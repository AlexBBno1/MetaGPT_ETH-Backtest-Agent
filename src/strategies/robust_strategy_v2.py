"""
RobustStrategy V2 - Reduced Parameters, Strict Regime Gate

Key Changes from V1:
1. Parameter reduction: <= 12 parameters (hard limit)
2. Strict regime gate: bad regime = FLAT (0% exposure)
3. Position change threshold: avoid micro-adjustments
4. Min hold enforcement + cooldown to control turnover
5. Vol-target sizing capped by leverage

Parameters (12 total):
1. donchian_entry_period: Entry lookback
2. donchian_exit_period: Exit lookback
3. adx_gate_threshold: ADX gate for trending
4. vol_target: Target volatility for sizing
5. leverage_cap: Max leverage
6. min_hold_hours: Minimum holding period
7. cooldown_hours: Cooldown after trade
8. position_change_threshold: Min change to adjust position
9. atr_stop_mult: ATR multiplier for stops
10. crash_vol_mult: Volatility multiplier for crash detection
11. chop_threshold: Choppiness threshold for regime
12. regime_ma_period: SMA period for regime detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from indicators import (
    sma, ema, atr, donchian, adx, choppiness_index, 
    realized_volatility
)


class RobustStrategyV2:
    """
    Robust Strategy V2 with strict regime gate and reduced parameters.
    
    Core Philosophy:
    - Simple is better: fewer parameters = less overfitting
    - Bad regime = FLAT: no "8% exposure" bleeding
    - Trend following only: no mean reversion components
    - Strict position management: avoid micro-adjustments
    """
    
    # Default parameters (12 total)
    # Tuned for mid-frequency turnover target ~80-120 trades/year
    DEFAULT_PARAMS = {
        'donchian_entry_period': 144,     # ~6 days
        'donchian_exit_period': 72,       # ~3 days
        'adx_gate_threshold': 20,         # Moderate trend filter
        'vol_target': 0.35,               # Target annual vol for sizing
        'leverage_cap': 1.6,              # Max leverage cap
        'min_hold_hours': 72,             # 3 days minimum hold
        'cooldown_hours': 24,             # 1 day cooldown
        'position_change_threshold': 0.20,# Only adjust if change is material
        'atr_stop_mult': 3.0,             # ATR-based stop
        'crash_vol_mult': 1.8,            # Crash detection multiplier
        'chop_threshold': 55,             # Chop filter
        'regime_ma_period': 120,          # Regime MA for direction bias
    }
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize with parameters."""
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)
        
        # Validate parameter count (exclude control params like 'disable_regime_gate')
        control_params = {'disable_regime_gate'}
        strategy_params = {k: v for k, v in self.params.items() if k not in control_params}
        assert len(strategy_params) <= 12, f"Too many parameters: {len(strategy_params)} > 12"
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators."""
        df = df.copy()
        
        p = self.params
        
        # Donchian channels for entry/exit
        df['don_entry_upper'], df['don_entry_lower'], _ = donchian(df, p['donchian_entry_period'])
        df['don_exit_upper'], df['don_exit_lower'], _ = donchian(df, p['donchian_exit_period'])
        
        # ADX for trend strength
        df['adx'] = adx(df, period=14)
        
        # Choppiness index
        df['chop'] = choppiness_index(df, period=14)
        
        # Regime SMA
        df['regime_ma'] = sma(df['close'], p['regime_ma_period'])
        
        # ATR for stops and sizing
        df['atr'] = atr(df, period=14)
        df['atr_pct'] = df['atr'] / df['close']
        
        # Realized volatility for sizing
        df['realized_vol'] = realized_volatility(df['close'], 20, annualize=True)
        
        # Crash detection
        vol_ma = df['realized_vol'].rolling(40).mean()
        df['is_crash'] = (df['realized_vol'] > vol_ma * p['crash_vol_mult']) & \
                         (df['close'].pct_change(24) < -0.05)
        
        return df
    
    def regime_pass(self, row: pd.Series) -> bool:
        """
        Binary regime gate: bad regime = FLAT (0% exposure).
        Requires trend strength (ADX) and low choppiness.
        
        If 'disable_regime_gate' param is True, only checks for crashes.
        """
        p = self.params
        
        # Always check for crashes
        if row.get('is_crash', False):
            return False
        
        # If regime gate is disabled, skip ADX/CHOP checks
        if p.get('disable_regime_gate', False):
            return True
        
        if pd.isna(row['adx']) or pd.isna(row['chop']):
            return False
        if row['adx'] < p['adx_gate_threshold']:
            return False
        if row['chop'] > p['chop_threshold']:
            return False
        return True
    
    def calculate_position_size(self, row: pd.Series, direction: int) -> float:
        """
        Volatility-targeted position sizing capped by leverage.
        Returns position from -leverage_cap to +leverage_cap.
        """
        if direction == 0:
            return 0.0
        
        p = self.params
        current_vol = max(row['realized_vol'], 1e-6)
        vol_scalar = p['vol_target'] / current_vol
        
        # Clip to avoid micro-sizing and respect leverage cap
        base_size = np.clip(vol_scalar, 0.2, p['leverage_cap'])
        position = direction * base_size
        return float(np.clip(position, -p['leverage_cap'], p['leverage_cap']))
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals (mid-frequency, turnover controlled).
        
        Logic:
        1) Regime gate: if failed or crash -> flat.
        2) Entry: Donchian breakout, only if cooldown passed and regime good.
        3) Exit: ATR stop or Donchian exit after min hold.
        4) Sizing: vol-targeted, capped by leverage; apply position-change threshold.
        """
        df = self.prepare_data(df)
        p = self.params
        
        n = len(df)
        signals = pd.Series(0.0, index=df.index)
        
        # State tracking
        position = 0.0
        position_direction = 0  # 1: long, -1: short, 0: flat
        bars_in_position = 0
        bars_since_exit = 9999  # Start with no cooldown
        entry_price = 0.0
        stop_price = 0.0
        
        for i in range(1, n):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]

            # Warm-up handling: skip if key indicators not ready
            needed = [
                'don_entry_upper', 'don_entry_lower', 'don_exit_upper', 'don_exit_lower',
                'adx', 'chop', 'atr', 'regime_ma', 'realized_vol'
            ]
            if any(pd.isna(row[k]) for k in needed):
                signals.iloc[i] = position
                continue
            
            # Update bar counters
            if position_direction != 0:
                bars_in_position += 1
            else:
                bars_since_exit += 1
            
            # === CRASH CHECK ===
            if row.get('is_crash', False) and position_direction != 0:
                position_direction = 0
                position = 0.0
                bars_in_position = 0
                bars_since_exit = 0
                stop_price = 0.0
                signals.iloc[i] = 0.0
                continue
            
            # === STOP LOSS CHECK ===
            if position_direction != 0 and stop_price > 0:
                if (position_direction == 1 and row['close'] < stop_price) or \
                   (position_direction == -1 and row['close'] > stop_price):
                    position_direction = 0
                    position = 0.0
                    bars_in_position = 0
                    bars_since_exit = 0
                    stop_price = 0.0
                    signals.iloc[i] = 0.0
                    continue
            
            # === DONCHIAN EXIT CHECK (after min hold) ===
            if position_direction != 0 and bars_in_position >= p['min_hold_hours']:
                exit_signal = False
                if position_direction == 1 and row['close'] < row['don_exit_lower']:
                    exit_signal = True
                elif position_direction == -1 and row['close'] > row['don_exit_upper']:
                    exit_signal = True
                
                if exit_signal:
                    position_direction = 0
                    position = 0.0
                    bars_in_position = 0
                    bars_since_exit = 0
                    stop_price = 0.0
                    signals.iloc[i] = 0.0
                    continue
            
            # === ENTRY / ADJUSTMENT LOGIC ===
            desired_direction = position_direction
            desired_position = position
            
            gate_ok = self.regime_pass(row)
            
            if not gate_ok:
                desired_direction = 0
                desired_position = 0.0
            else:
                # Entry only when flat and cooldown satisfied
                if position_direction == 0 and bars_since_exit >= p['cooldown_hours']:
                    if row['close'] > prev_row['don_entry_upper'] and row['close'] > row['regime_ma']:
                        desired_direction = 1
                        entry_price = row['close']
                        stop_price = entry_price - p['atr_stop_mult'] * row['atr']
                        bars_in_position = 0
                    elif row['close'] < prev_row['don_entry_lower'] and row['close'] < row['regime_ma']:
                        desired_direction = -1
                        entry_price = row['close']
                        stop_price = entry_price + p['atr_stop_mult'] * row['atr']
                        bars_in_position = 0
            
            # Position sizing (vol target) and change threshold
            if desired_direction == 0:
                desired_position = 0.0
            else:
                desired_position = self.calculate_position_size(row, desired_direction)
            
            if abs(desired_position - position) >= p['position_change_threshold'] or desired_position == 0.0:
                prev_direction = position_direction  # Track previous state
                position = desired_position
                position_direction = int(np.sign(position))
                if position_direction == 0:
                    bars_in_position = 0
                    # Only reset cooldown when actually EXITING a position (not when already flat)
                    if prev_direction != 0:
                        bars_since_exit = 0
                    stop_price = 0.0
            
            signals.iloc[i] = position
        
        return signals
    
    @staticmethod
    def get_param_bounds() -> Dict[str, Tuple]:
        """Get parameter bounds for optimization (constrained to <=12 params)."""
        return {
            'donchian_entry_period': (72, 264),    # 3-11 days
            'donchian_exit_period': (36, 168),     # 1.5-7 days
            'adx_gate_threshold': (12, 35),        # Trend strength filter (wider to avoid zero trades)
            'vol_target': (0.20, 0.55),            # Annual vol target
            'leverage_cap': (1.0, 2.0),            # Leverage range
            'min_hold_hours': (36, 144),           # 1.5-6 days min hold
            'cooldown_hours': (8, 96),             # 0.3-4 days cooldown
            'position_change_threshold': (0.10, 0.50),
            'atr_stop_mult': (2.0, 4.5),           # Tighter stops
            'crash_vol_mult': (1.5, 3.0),
            'chop_threshold': (40, 70),            # Chop filter wider
            'regime_ma_period': (80, 200),
        }
    
    @staticmethod
    def sample_params(trial, bounds: Dict[str, Tuple] = None) -> Dict[str, Any]:
        """Sample parameters for Optuna trial."""
        if bounds is None:
            bounds = RobustStrategyV2.get_param_bounds()
        
        params = {
            'donchian_entry_period': trial.suggest_int(
                'donchian_entry_period', bounds['donchian_entry_period'][0], bounds['donchian_entry_period'][1]
            ),
            'donchian_exit_period': trial.suggest_int(
                'donchian_exit_period', bounds['donchian_exit_period'][0], bounds['donchian_exit_period'][1]
            ),
            'adx_gate_threshold': trial.suggest_int(
                'adx_gate_threshold', bounds['adx_gate_threshold'][0], bounds['adx_gate_threshold'][1]
            ),
            'vol_target': trial.suggest_float(
                'vol_target', bounds['vol_target'][0], bounds['vol_target'][1]
            ),
            'leverage_cap': trial.suggest_float(
                'leverage_cap', bounds['leverage_cap'][0], bounds['leverage_cap'][1]
            ),
            'min_hold_hours': trial.suggest_int(
                'min_hold_hours', bounds['min_hold_hours'][0], bounds['min_hold_hours'][1]
            ),
            'cooldown_hours': trial.suggest_int(
                'cooldown_hours', bounds['cooldown_hours'][0], bounds['cooldown_hours'][1]
            ),
            'position_change_threshold': trial.suggest_float(
                'position_change_threshold', bounds['position_change_threshold'][0], bounds['position_change_threshold'][1]
            ),
            'atr_stop_mult': trial.suggest_float(
                'atr_stop_mult', bounds['atr_stop_mult'][0], bounds['atr_stop_mult'][1]
            ),
            'crash_vol_mult': trial.suggest_float(
                'crash_vol_mult', bounds['crash_vol_mult'][0], bounds['crash_vol_mult'][1]
            ),
            'chop_threshold': trial.suggest_int(
                'chop_threshold', bounds['chop_threshold'][0], bounds['chop_threshold'][1]
            ),
            'regime_ma_period': trial.suggest_int(
                'regime_ma_period', bounds['regime_ma_period'][0], bounds['regime_ma_period'][1]
            ),
        }
        
        return params


if __name__ == "__main__":
    from data_loader import load_data
    from backtester import run_backtest
    
    print("Testing RobustStrategyV2...")
    df = load_data(start_date="2023-01-01", end_date="2024-01-01", freq="1h")
    
    strategy = RobustStrategyV2()
    signals = strategy.generate_signals(df)
    
    print(f"Signal stats:")
    print(f"  Non-zero signals: {(signals != 0).sum()} / {len(signals)}")
    print(f"  Mean position: {signals.mean():.4f}")
    print(f"  Exposure: {(signals != 0).mean()*100:.1f}%")
    
    result = run_backtest(df, signals, leverage=1.0)
    
    print(f"\nBacktest Results:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")

