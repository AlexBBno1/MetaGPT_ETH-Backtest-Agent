"""
Robust Trend-Following Strategy with Validated Factors.

This strategy uses a simpler, more robust approach to reduce overfitting:
1. Trend-following with ADX filter
2. Simple regime-based position sizing
3. Volatility-based risk management
4. Crash protection via momentum filter
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from indicators import sma, ema, atr, adx, rsi, donchian, bollinger_bands, realized_volatility


class RobustTrendStrategy:
    """
    Robust trend-following strategy optimized for ETH.
    
    Key principles:
    1. Follow the trend (momentum + moving averages)
    2. Filter by trend strength (ADX)
    3. Adjust position by volatility regime
    4. Protect against crashes
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'RobustTrendStrategy')
        self.leverage = config.get('leverage', 1.0)
        self.max_leverage = config.get('max_leverage', 2.0)
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators."""
        df = df.copy()
        
        # Get parameters from config
        params = self.config.get('params', {})
        
        # Moving averages
        fast_ma = params.get('fast_ma', 20)
        slow_ma = params.get('slow_ma', 50)
        trend_ma = params.get('trend_ma', 100)
        
        df['ema_fast'] = ema(df['close'], fast_ma)
        df['ema_slow'] = ema(df['close'], slow_ma)
        df['sma_trend'] = sma(df['close'], trend_ma)
        
        # ADX for trend strength
        adx_period = params.get('adx_period', 14)
        df['adx'] = adx(df, adx_period)
        
        # ATR for volatility
        atr_period = params.get('atr_period', 14)
        df['atr'] = atr(df, atr_period)
        df['atr_pct'] = df['atr'] / df['close']
        
        # Volatility regime
        vol_period = params.get('vol_period', 20)
        df['realized_vol'] = realized_volatility(df['close'], vol_period, annualize=True)
        df['vol_ma'] = df['realized_vol'].rolling(window=vol_period * 2).mean()
        
        # Momentum
        mom_period = params.get('momentum_period', 24)
        df['momentum'] = df['close'].pct_change(mom_period)
        
        # RSI for crash protection
        df['rsi'] = rsi(df['close'], 14)
        
        # Donchian Channel
        don_period = params.get('donchian_period', 20)
        df['don_high'], df['don_low'], df['don_mid'] = donchian(df, don_period)
        
        return df
    
    def generate_trend_signal(self, df: pd.DataFrame) -> pd.Series:
        """Generate base trend signal from moving average crossover."""
        # EMA crossover
        ema_signal = np.sign(df['ema_fast'] - df['ema_slow'])
        
        # Trend alignment (price above/below long-term MA)
        trend_align = np.sign(df['close'] - df['sma_trend'])
        
        # Donchian breakout
        don_signal = pd.Series(0.0, index=df.index)
        don_signal[df['close'] > df['don_high'].shift(1)] = 1.0
        don_signal[df['close'] < df['don_low'].shift(1)] = -1.0
        
        # Combined signal: need 2 out of 3 to agree
        combined = (ema_signal + trend_align + don_signal) / 3.0
        base_signal = np.sign(combined)
        
        return base_signal
    
    def apply_adx_filter(self, df: pd.DataFrame, signal: pd.Series) -> pd.Series:
        """Filter signals by ADX trend strength."""
        params = self.config.get('params', {})
        adx_threshold = params.get('adx_threshold', 20)
        
        # Reduce signal when ADX is low (no trend)
        adx_mult = pd.Series(1.0, index=df.index)
        adx_mult[df['adx'] < adx_threshold] = 0.3
        adx_mult[df['adx'] < adx_threshold * 0.7] = 0.0
        
        return signal * adx_mult
    
    def apply_vol_sizing(self, df: pd.DataFrame, signal: pd.Series) -> pd.Series:
        """Adjust position size based on volatility."""
        params = self.config.get('params', {})
        target_vol = params.get('target_vol', 0.5)
        
        # Inverse volatility scaling (vol targeting)
        vol_safe = df['realized_vol'].replace(0, np.nan).fillna(df['realized_vol'].mean())
        vol_ratio = target_vol / vol_safe
        vol_ratio = vol_ratio.clip(0.5, 2.0)  # Cap scaling
        
        # Also reduce in very high vol environments
        high_vol = df['realized_vol'] > df['vol_ma'] * 2.0
        vol_ratio[high_vol] = vol_ratio[high_vol] * 0.5
        
        return signal * vol_ratio
    
    def apply_crash_protection(self, df: pd.DataFrame, signal: pd.Series) -> pd.Series:
        """Reduce exposure during potential crashes."""
        # Detect crash conditions:
        # 1. Sharp negative momentum
        # 2. High volatility spike
        # 3. Extreme RSI
        
        crash_momentum = df['momentum'] < -0.1  # >10% drop
        high_vol_spike = df['realized_vol'] > df['vol_ma'] * 2.5
        extreme_rsi = (df['rsi'] < 25) | (df['rsi'] > 85)
        
        # Progressive reduction
        protection = pd.Series(1.0, index=df.index)
        protection[crash_momentum] = 0.5
        protection[high_vol_spike] = 0.5
        protection[extreme_rsi] = 0.7
        protection[crash_momentum & high_vol_spike] = 0.2
        
        # Smooth protection to avoid whipsaws
        protection = protection.rolling(window=4).mean().fillna(1.0)
        
        return signal * protection
    
    def apply_holding_period(self, signal: pd.Series, min_hold: int = 8) -> pd.Series:
        """Enforce minimum holding period."""
        result = signal.copy()
        last_change = 0
        current_pos = 0.0
        
        for i in range(len(result)):
            if np.sign(signal.iloc[i]) != np.sign(current_pos):
                if i - last_change >= min_hold or abs(current_pos) < 0.01:
                    current_pos = signal.iloc[i]
                    last_change = i
            result.iloc[i] = current_pos
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate final trading signals."""
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Base trend signal
        signal = self.generate_trend_signal(df)
        
        # Apply filters and adjustments
        signal = self.apply_adx_filter(df, signal)
        signal = self.apply_vol_sizing(df, signal)
        signal = self.apply_crash_protection(df, signal)
        
        # Enforce holding period
        params = self.config.get('params', {})
        min_hold = params.get('min_hold', 8)
        signal = self.apply_holding_period(signal, min_hold)
        
        # Apply base position sizing
        base_size = self.config.get('base_position_size', 0.8)
        signal = signal * base_size
        
        # Clip final signal
        signal = signal.clip(-1.0, 1.0)
        
        return signal
    
    @staticmethod
    def get_route_config(route: str) -> Dict:
        """Get configuration for specific route."""
        
        # Common parameters
        base_params = {
            'fast_ma': 20,
            'slow_ma': 50,
            'trend_ma': 100,
            'adx_period': 14,
            'atr_period': 14,
            'vol_period': 20,
            'momentum_period': 24,
            'donchian_period': 20,
        }
        
        configs = {
            'A': {
                'name': 'Route A (Conservative)',
                'leverage': 1.0,
                'max_leverage': 1.2,
                'base_position_size': 0.7,
                'params': {
                    **base_params,
                    'adx_threshold': 25,
                    'target_vol': 0.35,
                    'min_hold': 12,
                },
            },
            'B': {
                'name': 'Route B (Balanced)',
                'leverage': 1.3,
                'max_leverage': 1.5,
                'base_position_size': 0.85,
                'params': {
                    **base_params,
                    'adx_threshold': 22,
                    'target_vol': 0.45,
                    'min_hold': 8,
                },
            },
            'C': {
                'name': 'Route C (Aggressive)',
                'leverage': 1.8,
                'max_leverage': 2.2,
                'base_position_size': 1.0,
                'params': {
                    **base_params,
                    'adx_threshold': 18,
                    'target_vol': 0.60,
                    'min_hold': 6,
                },
            },
        }
        
        return configs.get(route.upper(), configs['A'])


if __name__ == "__main__":
    from data_loader import load_data
    from backtester import Backtester, BacktestConfig
    
    # Test
    print("Loading data...")
    df = load_data(start_date="2020-01-01", end_date="2025-11-30", freq="1h")
    
    for route in ['A', 'B', 'C']:
        config = RobustTrendStrategy.get_route_config(route)
        strategy = RobustTrendStrategy(config)
        signals = strategy.generate_signals(df)
        
        bt_config = BacktestConfig(
            initial_capital=15000.0,
            leverage=config['leverage'],
            max_leverage=config['max_leverage']
        )
        
        backtester = Backtester(bt_config)
        result = backtester.run(df, signals)
        
        print(f"\n{config['name']}:")
        print(f"  CAGR: {result.metrics['CAGR']:.2f}%")
        print(f"  MaxDD: {result.metrics['MaxDrawdown']:.2f}%")
        print(f"  Sharpe: {result.metrics['Sharpe']:.2f}")
        print(f"  Final: ${result.metrics['FinalEquity']:,.0f}")

