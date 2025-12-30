"""
Factor-Based Strategy combining validated factors from Factor Exploration.

Factors included:
1. Trend Persistence (ADX + Hurst)
2. Chop Detector
3. Rules-Based Regime Sizing
4. Volatility Breakout
5. Crash Protection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indicators import (
    sma, ema, rsi, atr, donchian, bollinger_bands,
    adx, trend_persistence, chop_detector, choppiness_index,
    rules_regime, regime_position_sizing,
    volatility_breakout_signal, bollinger_squeeze,
    crash_protection_signal,
    calculate_all_factors
)


class FactorStrategy:
    """
    Multi-factor strategy using validated factors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy with route configuration.
        
        Args:
            config: Route configuration dict
        """
        self.config = config
        self.name = config.get('name', 'FactorStrategy')
        
        # Extract key parameters
        self.leverage = config.get('leverage', 1.0)
        self.max_leverage = config.get('max_leverage', 2.0)
        self.base_position_size = config.get('base_position_size', 1.0)
        self.factor_weights = config.get('factor_weights', {})
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data with all required indicators and factors.
        """
        # Build params dict from config
        params = {}
        
        # Trend persistence params
        tp_params = self.config.get('trend_persistence_params', {})
        params['adx_period'] = tp_params.get('adx_period', 14)
        params['adx_threshold'] = tp_params.get('adx_threshold', 25)
        params['hurst_period'] = tp_params.get('hurst_period', 20)
        
        # Chop filter params
        chop_params = self.config.get('chop_filter_params', {})
        params['chop_period'] = chop_params.get('chop_period', 14)
        params['chop_threshold'] = chop_params.get('chop_threshold', 60)
        
        # Regime params
        regime_params = self.config.get('regime_params', {})
        params['regime_adx_period'] = regime_params.get('adx_period', 14)
        params['regime_sma_period'] = regime_params.get('sma_period', 50)
        
        # Vol breakout params
        vb_params = self.config.get('vol_breakout_params', {})
        params['vol_bb_period'] = vb_params.get('bb_period', 20)
        params['vol_bb_std'] = vb_params.get('bb_std', 2.0)
        
        # Crash protection params
        cp_params = self.config.get('crash_protection_params', {})
        params['crash_vol_lookback'] = cp_params.get('vol_lookback', 20)
        params['crash_vol_mult'] = cp_params.get('crash_vol_mult', 2.0)
        params['crash_momentum_lookback'] = cp_params.get('momentum_lookback', 24)
        
        # Signal params
        sig_params = self.config.get('signal_params', {})
        params['donchian_period'] = sig_params.get('donchian_period', 20)
        
        # Calculate all factors
        df = calculate_all_factors(df, params)
        
        return df
    
    def generate_base_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate base trading signal from momentum and trend indicators.
        Returns raw signal before factor adjustments.
        """
        sig_params = self.config.get('signal_params', {})
        
        momentum_lookback = sig_params.get('momentum_lookback', 24)
        ema_fast = sig_params.get('ema_fast', 12)
        ema_slow = sig_params.get('ema_slow', 26)
        donchian_period = sig_params.get('donchian_period', 20)
        
        # Calculate EMAs if not present
        if f'ema_{ema_fast}' not in df.columns:
            df[f'ema_{ema_fast}'] = ema(df['close'], ema_fast)
        if f'ema_{ema_slow}' not in df.columns:
            df[f'ema_{ema_slow}'] = ema(df['close'], ema_slow)
        
        # Momentum signal
        momentum = df['close'].pct_change(momentum_lookback)
        momentum_signal = np.sign(momentum)
        
        # EMA crossover signal
        ema_diff = df[f'ema_{ema_fast}'] - df[f'ema_{ema_slow}']
        ema_signal = np.sign(ema_diff)
        
        # Donchian breakout signal
        don_upper, don_lower, _ = donchian(df, donchian_period)
        don_signal = pd.Series(0, index=df.index)
        don_signal[df['close'] > don_upper.shift(1)] = 1
        don_signal[df['close'] < don_lower.shift(1)] = -1
        
        # Combined base signal (average of signals)
        base_signal = (momentum_signal + ema_signal + don_signal) / 3.0
        
        return base_signal
    
    def apply_trend_persistence_filter(self, df: pd.DataFrame, signal: pd.Series) -> pd.Series:
        """
        Apply trend persistence filter to reduce signals in non-trending markets.
        """
        weight = self.factor_weights.get('trend_persistence', 0)
        if weight == 0:
            return signal
        
        tp_params = self.config.get('trend_persistence_params', {})
        adx_threshold = tp_params.get('adx_threshold', 25)
        
        # Get trend persistence score (0 to 1)
        tp_score = df['trend_persistence']
        
        # Only keep full signal when ADX > threshold
        adx_val = df['adx']
        signal_mult = pd.Series(1.0, index=df.index)
        
        # Reduce signal in low trend environments
        signal_mult[adx_val < adx_threshold] = 0.5
        signal_mult[tp_score < 0.3] = 0.3
        
        return signal * signal_mult
    
    def apply_chop_filter(self, df: pd.DataFrame, signal: pd.Series) -> pd.Series:
        """
        Apply chop detector to avoid trading in ranging markets.
        """
        weight = self.factor_weights.get('chop_filter', 0)
        if weight == 0:
            return signal
        
        chop_params = self.config.get('chop_filter_params', {})
        chop_threshold = chop_params.get('chop_threshold', 60)
        
        chop_index = df['chop_index']
        
        # Reduce signal in choppy markets
        signal_mult = pd.Series(1.0, index=df.index)
        signal_mult[chop_index > chop_threshold] = 0.3
        signal_mult[chop_index > chop_threshold + 10] = 0.1
        
        return signal * signal_mult
    
    def apply_regime_sizing(self, df: pd.DataFrame, signal: pd.Series) -> pd.Series:
        """
        Apply regime-based position sizing.
        """
        weight = self.factor_weights.get('regime_sizing', 0)
        if weight == 0:
            return signal
        
        regime_params = self.config.get('regime_params', {})
        regime_multipliers = regime_params.get('regime_multipliers', {
            0: 0.5, 1: 1.0, -1: 1.0, 2: 0.7
        })
        
        regime = df['regime']
        
        return regime_position_sizing(regime, signal, regime_multipliers)
    
    def apply_vol_breakout(self, df: pd.DataFrame, signal: pd.Series) -> pd.Series:
        """
        Add volatility breakout component to signal.
        """
        weight = self.factor_weights.get('vol_breakout', 0)
        if weight == 0:
            return signal
        
        vb_signal = df['vol_breakout_signal']
        
        # Blend breakout signal with base signal
        combined = signal * (1 - weight) + vb_signal * weight
        
        return combined
    
    def apply_crash_protection(self, df: pd.DataFrame, signal: pd.Series) -> pd.Series:
        """
        Apply crash protection to reduce exposure during market stress.
        """
        weight = self.factor_weights.get('crash_protection', 0)
        if weight == 0:
            return signal
        
        protection_mult = df['crash_protection']
        
        # Apply protection (multiplier is 0-1)
        return signal * protection_mult
    
    def apply_holding_period(self, signal: pd.Series, min_hold: int = 6) -> pd.Series:
        """
        Enforce minimum holding period to reduce turnover.
        """
        result = signal.copy()
        last_change_idx = 0
        current_pos = 0.0
        
        for i in range(len(result)):
            bars_held = i - last_change_idx
            
            if abs(signal.iloc[i] - current_pos) > 0.1:  # Significant change
                if bars_held >= min_hold or abs(current_pos) < 0.01:
                    current_pos = signal.iloc[i]
                    last_change_idx = i
            
            result.iloc[i] = current_pos
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate final trading signals incorporating all factors.
        
        Returns:
            Series of target positions from -1 to +1
        """
        # Prepare data with all factors
        df = self.prepare_data(df)
        
        # Generate base signal
        signal = self.generate_base_signal(df)
        
        # Apply factors in sequence
        signal = self.apply_trend_persistence_filter(df, signal)
        signal = self.apply_chop_filter(df, signal)
        signal = self.apply_regime_sizing(df, signal)
        signal = self.apply_vol_breakout(df, signal)
        signal = self.apply_crash_protection(df, signal)
        
        # Apply position sizing
        signal = signal * self.base_position_size
        
        # Apply minimum holding period
        sig_params = self.config.get('signal_params', {})
        min_hold = sig_params.get('holding_period_min', 6)
        signal = self.apply_holding_period(signal, min_hold)
        
        # Clip to valid range
        signal = signal.clip(-1, 1)
        
        return signal
    
    @staticmethod
    def optimize_params(trial, route_config: dict) -> dict:
        """
        Generate optimized parameters using Optuna trial.
        """
        opt_bounds = route_config.get('opt_bounds', {})
        
        params = {}
        
        # Leverage
        lev_bounds = opt_bounds.get('leverage', (1.0, 2.0))
        params['leverage'] = trial.suggest_float('leverage', lev_bounds[0], lev_bounds[1])
        
        # ADX threshold
        adx_bounds = opt_bounds.get('adx_threshold', (20, 40))
        params['adx_threshold'] = trial.suggest_int('adx_threshold', adx_bounds[0], adx_bounds[1])
        
        # Chop threshold
        chop_bounds = opt_bounds.get('chop_threshold', (50, 70))
        params['chop_threshold'] = trial.suggest_int('chop_threshold', chop_bounds[0], chop_bounds[1])
        
        # Momentum lookback
        mom_bounds = opt_bounds.get('momentum_lookback', (12, 72))
        params['momentum_lookback'] = trial.suggest_int('momentum_lookback', mom_bounds[0], mom_bounds[1])
        
        # EMA periods
        params['ema_fast'] = trial.suggest_int('ema_fast', 8, 20)
        params['ema_slow'] = trial.suggest_int('ema_slow', 18, 50)
        
        # Donchian period
        params['donchian_period'] = trial.suggest_int('donchian_period', 15, 40)
        
        return params
    
    def update_config(self, params: dict):
        """
        Update configuration with optimized parameters.
        """
        if 'leverage' in params:
            self.config['leverage'] = params['leverage']
            self.leverage = params['leverage']
        
        if 'adx_threshold' in params:
            self.config.setdefault('trend_persistence_params', {})['adx_threshold'] = params['adx_threshold']
        
        if 'chop_threshold' in params:
            self.config.setdefault('chop_filter_params', {})['chop_threshold'] = params['chop_threshold']
        
        if 'momentum_lookback' in params:
            self.config.setdefault('signal_params', {})['momentum_lookback'] = params['momentum_lookback']
        
        if 'ema_fast' in params:
            self.config.setdefault('signal_params', {})['ema_fast'] = params['ema_fast']
        
        if 'ema_slow' in params:
            self.config.setdefault('signal_params', {})['ema_slow'] = params['ema_slow']
        
        if 'donchian_period' in params:
            self.config.setdefault('signal_params', {})['donchian_period'] = params['donchian_period']


if __name__ == "__main__":
    from data_loader import load_data
    from backtester import run_backtest
    from route_configs import ROUTE_A_CONFIG, ROUTE_B_CONFIG, ROUTE_C_CONFIG
    
    # Test with Route A
    print("Testing Factor Strategy with Route A...")
    df = load_data(start_date="2023-01-01", end_date="2024-01-01", freq="1h")
    
    strategy = FactorStrategy(ROUTE_A_CONFIG)
    signals = strategy.generate_signals(df)
    
    result = run_backtest(df, signals, leverage=strategy.leverage)
    
    print(f"\nRoute A Results:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")

