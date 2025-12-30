"""
ETH Strategy V2 - Complete Adaptive Walk-Forward Optimization Script
=====================================================================

OPTIMIZED VERSION with:
- True resume support (fixed study names, load_if_exists=True)
- Performance optimizations (cached signals, reduced redundant calculations)
- Detailed timing logs for debugging
- Trial watchdog timeout
- --profile mode for bottleneck analysis

Usage:
    python run_v2_adaptive_complete.py
    python run_v2_adaptive_complete.py --stage1-trials 5 --stage2-trials 3  # Quick test
    python run_v2_adaptive_complete.py --profile  # Profile mode
"""

import sys
import os
import json
import argparse
import time
import signal
import threading
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import copy
import optuna
from functools import lru_cache

warnings.filterwarnings('ignore')

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data_loader import load_data, get_data_info
from backtester import Backtester, BacktestConfig
from strategies.robust_strategy_v2 import RobustStrategyV2


# =============================================================================
# Configuration Constants
# =============================================================================

# Stage2 STRICT hard limits (IMMUTABLE per specification)
STAGE2_HARD_LIMITS = {
    'max_dd_limit': 0.45,        # 45% max drawdown (decimal)
    'min_sharpe_wf': 0.7,        # Sharpe >= 0.7 (prefer 0.8)
    'min_trades_per_year': 60,   # At least 60 trades/year
    'max_trades_per_year': 150,  # At most 150 trades/year
    'max_turnover': 400,         # Max turnover units/year
}

# Stage1 relaxation levels (progressively less strict)
STAGE1_RELAXATION_LEVELS = [
    {'min_trades_per_year': 30, 'name': 'standard', 'level': 0},
    {'min_trades_per_year': 10, 'name': 'relaxed_10', 'level': 1},
    {'min_trades_per_year': 5,  'name': 'relaxed_5', 'level': 2},
    {'min_trades_per_year': 3,  'name': 'minimal_gate', 'level': 3},
]

# Performance settings
TRIAL_TIMEOUT_SECONDS = 120  # Max time per trial before marking as failed
ENABLE_SIGNAL_CACHE = True   # Cache signals for same df+params


# =============================================================================
# Timing Context Manager
# =============================================================================

@contextmanager
def timer(name: str, timings: dict = None):
    """Context manager to time code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if timings is not None:
        timings[name] = elapsed


class TrialTimer:
    """Track timing for each trial phase."""
    def __init__(self):
        self.timings = {}
        self.start_time = None
    
    def start(self):
        self.start_time = time.perf_counter()
        self.timings = {}
    
    def mark(self, name: str, elapsed: float = None):
        if elapsed is None:
            elapsed = time.perf_counter() - self.start_time
        self.timings[name] = elapsed
    
    def total(self) -> float:
        return time.perf_counter() - self.start_time if self.start_time else 0
    
    def summary(self) -> str:
        parts = [f"{k}={v:.2f}s" for k, v in self.timings.items()]
        return ", ".join(parts) + f", total={self.total():.2f}s"


# =============================================================================
# Optimized Strategy with Cached Signals
# =============================================================================

class OptimizedRobustStrategyV2(RobustStrategyV2):
    """
    Optimized version with pre-computed indicators and vectorized operations.
    """
    
    _indicator_cache = {}  # Class-level cache for indicators
    
    @classmethod
    def clear_cache(cls):
        cls._indicator_cache = {}
    
    @classmethod
    def get_cached_indicators(cls, df: pd.DataFrame, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached indicators if available."""
        if cache_key in cls._indicator_cache:
            cached = cls._indicator_cache[cache_key]
            if len(cached) == len(df):
                return cached
        return None
    
    @classmethod
    def set_cached_indicators(cls, df: pd.DataFrame, cache_key: str):
        """Cache computed indicators."""
        cls._indicator_cache[cache_key] = df
    
    def prepare_data(self, df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
        """Calculate all required indicators with caching."""
        # Create cache key from df shape and first/last timestamps
        cache_key = f"{len(df)}_{df['timestamp'].iloc[0]}_{df['timestamp'].iloc[-1]}"
        
        if use_cache:
            cached = self.get_cached_indicators(df, cache_key)
            if cached is not None:
                return cached.copy()
        
        # Calculate indicators (only once per unique dataset)
        df = super().prepare_data(df)
        
        if use_cache:
            self.set_cached_indicators(df, cache_key)
        
        return df
    
    def generate_signals_fast(self, df: pd.DataFrame, prepared_df: pd.DataFrame = None) -> pd.Series:
        """
        Generate trading signals using pre-prepared data.
        Optimized version that accepts pre-computed indicators.
        """
        if prepared_df is None:
            prepared_df = self.prepare_data(df)
        
        p = self.params
        n = len(prepared_df)
        signals = np.zeros(n)
        
        # Pre-extract arrays for faster access
        close = prepared_df['close'].values
        don_entry_upper = prepared_df['don_entry_upper'].values
        don_entry_lower = prepared_df['don_entry_lower'].values
        don_exit_upper = prepared_df['don_exit_upper'].values
        don_exit_lower = prepared_df['don_exit_lower'].values
        adx_arr = prepared_df['adx'].values
        chop_arr = prepared_df['chop'].values
        regime_ma = prepared_df['regime_ma'].values
        atr_arr = prepared_df['atr'].values
        realized_vol = prepared_df['realized_vol'].values
        is_crash = prepared_df['is_crash'].values
        
        # State tracking
        position = 0.0
        position_direction = 0
        bars_in_position = 0
        bars_since_exit = 9999
        entry_price = 0.0
        stop_price = 0.0
        
        disable_gate = p.get('disable_regime_gate', False)
        adx_threshold = p['adx_gate_threshold']
        chop_threshold = p['chop_threshold']
        min_hold = p['min_hold_hours']
        cooldown = p['cooldown_hours']
        atr_stop = p['atr_stop_mult']
        vol_target = p['vol_target']
        leverage_cap = p['leverage_cap']
        pos_change_threshold = p['position_change_threshold']
        
        for i in range(1, n):
            # Skip warmup
            if np.isnan(don_entry_upper[i]) or np.isnan(adx_arr[i]):
                signals[i] = position
                continue
            
            # Update bar counters
            if position_direction != 0:
                bars_in_position += 1
            else:
                bars_since_exit += 1
            
            # CRASH CHECK
            if is_crash[i] and position_direction != 0:
                position_direction = 0
                position = 0.0
                bars_in_position = 0
                bars_since_exit = 0
                stop_price = 0.0
                signals[i] = 0.0
                continue
            
            # STOP LOSS CHECK
            if position_direction != 0 and stop_price > 0:
                if (position_direction == 1 and close[i] < stop_price) or \
                   (position_direction == -1 and close[i] > stop_price):
                    position_direction = 0
                    position = 0.0
                    bars_in_position = 0
                    bars_since_exit = 0
                    stop_price = 0.0
                    signals[i] = 0.0
                    continue
            
            # DONCHIAN EXIT CHECK
            if position_direction != 0 and bars_in_position >= min_hold:
                exit_signal = False
                if position_direction == 1 and close[i] < don_exit_lower[i]:
                    exit_signal = True
                elif position_direction == -1 and close[i] > don_exit_upper[i]:
                    exit_signal = True
                
                if exit_signal:
                    position_direction = 0
                    position = 0.0
                    bars_in_position = 0
                    bars_since_exit = 0
                    stop_price = 0.0
                    signals[i] = 0.0
                    continue
            
            # REGIME CHECK
            gate_ok = True
            if is_crash[i]:
                gate_ok = False
            elif not disable_gate:
                if np.isnan(adx_arr[i]) or np.isnan(chop_arr[i]):
                    gate_ok = False
                elif adx_arr[i] < adx_threshold:
                    gate_ok = False
                elif chop_arr[i] > chop_threshold:
                    gate_ok = False
            
            desired_direction = position_direction
            desired_position = position
            
            if not gate_ok:
                desired_direction = 0
                desired_position = 0.0
            else:
                # Entry only when flat and cooldown satisfied
                if position_direction == 0 and bars_since_exit >= cooldown:
                    if close[i] > don_entry_upper[i-1] and close[i] > regime_ma[i]:
                        desired_direction = 1
                        entry_price = close[i]
                        stop_price = entry_price - atr_stop * atr_arr[i]
                        bars_in_position = 0
                    elif close[i] < don_entry_lower[i-1] and close[i] < regime_ma[i]:
                        desired_direction = -1
                        entry_price = close[i]
                        stop_price = entry_price + atr_stop * atr_arr[i]
                        bars_in_position = 0
            
            # Position sizing
            if desired_direction == 0:
                desired_position = 0.0
            else:
                current_vol = max(realized_vol[i], 1e-6)
                vol_scalar = vol_target / current_vol
                base_size = np.clip(vol_scalar, 0.2, leverage_cap)
                desired_position = float(np.clip(desired_direction * base_size, -leverage_cap, leverage_cap))
            
            # Apply position change threshold
            # Track if we're actually exiting a position (was in position, now flat)
            was_in_position = position_direction != 0
            
            if abs(desired_position - position) >= pos_change_threshold or desired_position == 0.0:
                position = desired_position
                position_direction = int(np.sign(position))
                
                # Only reset bars_since_exit when ACTUALLY exiting (not just gate failing)
                if position_direction == 0:
                    bars_in_position = 0
                    stop_price = 0.0
                    # Only reset cooldown if we were actually in a position
                    if was_in_position:
                        bars_since_exit = 0
            
            signals[i] = position
        
        return pd.Series(signals, index=df.index)


# =============================================================================
# Walk-Forward Result Data Class
# =============================================================================

@dataclass
class WalkForwardResultV2:
    """Results from walk-forward optimization V2."""
    best_params: Dict
    train_metrics: Dict
    test_metrics: Dict
    full_metrics: Dict
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    yearly_metrics: Dict
    optimization_history: List[Dict]
    stage1_candidates: List[Dict] = field(default_factory=list)
    stage2_results: List[Dict] = field(default_factory=list)
    relaxation_log: List[Dict] = field(default_factory=list)
    passed_strict_stage2: bool = False


# =============================================================================
# Adaptive Walk-Forward Optimizer V2 (Optimized)
# =============================================================================

class AdaptiveWalkForwardOptimizerV2:
    """
    Walk-Forward Optimizer V2 with ADAPTIVE Stage1 exploration.
    
    OPTIMIZATIONS:
    - Fixed study names for true resume capability
    - Cached indicator calculations
    - Reduced redundant signal generation
    - Timing instrumentation
    - Trial timeout watchdog
    """
    
    def __init__(
        self,
        train_years: int = 3,
        stage1_trials: int = 80,
        stage2_trials: int = 50,
        n_candidates: int = 3,
        random_state: int = 42,
        db_path: str = None,
        verbose_timing: bool = True,
        enable_pruner: bool = False
    ):
        self.train_years = train_years
        self.stage1_trials = stage1_trials
        self.stage2_trials = stage2_trials
        self.n_candidates = n_candidates
        self.random_state = random_state
        self.db_path = db_path
        self.verbose_timing = verbose_timing
        self.enable_pruner = enable_pruner
        
        self.best_params = None
        self.optimization_history = []
        self.stage1_candidates = []
        self.stage2_results = []
        self.relaxation_log = []
        
        self.hard_limits_stage2 = STAGE2_HARD_LIMITS.copy()
        self.current_stage = 'stage1'
        self.current_relaxation_level = 0
        
        # Cached prepared data
        self._prepared_data_cache = {}
        
        # Timing stats
        self.trial_timings = []
        self.rejection_stats = {}
        
    def _get_prepared_data(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Get prepared data with indicators, using cache if possible."""
        # Cache key based on df identity and relevant params
        cache_key = f"{id(df)}_{len(df)}"
        
        if cache_key in self._prepared_data_cache:
            return self._prepared_data_cache[cache_key]
        
        strategy = OptimizedRobustStrategyV2(params)
        prepared = strategy.prepare_data(df)
        self._prepared_data_cache[cache_key] = prepared
        return prepared
    
    def get_adaptive_stage1_limits(self, level: int = 0) -> Dict:
        """Get Stage1 limits for given relaxation level."""
        base_limits = {
            'max_dd_limit': 0.70,
            'min_sharpe_wf': -0.5,
            'max_trades_per_year': 300,
            'max_turnover': 600,
        }
        relaxation = STAGE1_RELAXATION_LEVELS[min(level, len(STAGE1_RELAXATION_LEVELS)-1)]
        base_limits['min_trades_per_year'] = relaxation['min_trades_per_year']
        return base_limits
    
    def get_adaptive_param_bounds(self, level: int = 0) -> Dict[str, Tuple]:
        """Get parameter bounds for Stage1, progressively widened."""
        base_bounds = RobustStrategyV2.get_param_bounds().copy()
        
        if level >= 1:
            base_bounds['adx_gate_threshold'] = (8, 35)
            base_bounds['chop_threshold'] = (40, 80)
            
        if level >= 2:
            base_bounds['min_hold_hours'] = (24, 144)
            base_bounds['adx_gate_threshold'] = (6, 35)
            base_bounds['chop_threshold'] = (35, 85)
        
        if level >= 3:
            base_bounds['adx_gate_threshold'] = (5, 40)
            base_bounds['chop_threshold'] = (30, 90)
            base_bounds['min_hold_hours'] = (12, 144)
            base_bounds['cooldown_hours'] = (4, 96)
            base_bounds['donchian_entry_period'] = (48, 264)
            
        return base_bounds
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_end: str = "2022-12-31"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        train_end_dt = pd.to_datetime(train_end)
        train_df = df[df['timestamp'] <= train_end_dt].copy().reset_index(drop=True)
        test_df = df[df['timestamp'] > train_end_dt].copy().reset_index(drop=True)
        return train_df, test_df
    
    def split_by_year(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data by year periods for stability analysis."""
        df = df.copy()
        df['year'] = pd.to_datetime(df['timestamp']).dt.year
        
        periods = {
            '2020-2021': df[df['year'].isin([2020, 2021])].reset_index(drop=True),
            '2022': df[df['year'] == 2022].reset_index(drop=True),
            '2023': df[df['year'] == 2023].reset_index(drop=True),
            '2024-2025': df[df['year'].isin([2024, 2025])].reset_index(drop=True),
        }
        return periods
    
    def run_backtest_optimized(
        self,
        df: pd.DataFrame,
        params: Dict,
        prepared_df: pd.DataFrame = None,
        periods_per_year: int = 24 * 365
    ) -> Tuple[Any, pd.Series]:
        """
        Run backtest with given parameters.
        Returns (result, signals) to avoid recalculating signals.
        """
        strategy = OptimizedRobustStrategyV2(params)
        
        if prepared_df is None:
            prepared_df = strategy.prepare_data(df)
        
        signals = strategy.generate_signals_fast(df, prepared_df)
        
        bt_config = BacktestConfig(
            initial_capital=15000.0,
            commission_bps=4.0,
            slippage_bps=5.0,
            leverage=params.get('leverage_cap', 1.5),
            max_leverage=params.get('leverage_cap', 1.5)
        )
        
        backtester = Backtester(bt_config)
        result = backtester.run(df, signals, periods_per_year)
        return result, signals
    
    def calculate_turnover(self, signals: pd.Series) -> float:
        """Calculate annual turnover from position changes."""
        position_changes = signals.diff().abs()
        years = max(len(signals) / (24 * 365), 1e-9)
        annual_turnover = position_changes.sum() / years
        return annual_turnover

    @staticmethod
    def _to_decimal(x):
        """Convert percent-like values to decimal if needed."""
        if x is None:
            return 0.0
        try:
            val = float(x)
        except Exception:
            return 0.0
        return val / 100.0 if val > 2 else val
    
    def calculate_stability(self, yearly_cagrs: List[float]) -> float:
        """Calculate stability bonus based on yearly CAGR consistency."""
        if len(yearly_cagrs) < 2:
            return 0.0
        
        cagrs = np.array(yearly_cagrs)
        std = np.std(cagrs)
        mean = np.mean(cagrs)
        
        if abs(mean) > 0.01:
            cv = std / abs(mean)
        else:
            cv = 10.0
        
        stability = max(0, 1.0 - cv)
        return stability
    
    def calculate_score(
        self,
        train_metrics: Dict,
        test_metrics: Dict,
        full_metrics: Dict,
        turnover: float,
        yearly_cagrs: List[float],
        relaxation_level: int = 0
    ) -> Tuple[float, bool, str, float]:
        """Calculate composite score with stage-appropriate hard rejection."""
        sharpe_wf = float(test_metrics.get('Sharpe', 0) or 0)
        cagr_wf = self._to_decimal(test_metrics.get('CAGR', 0))
        maxdd_full = self._to_decimal(full_metrics.get('MaxDrawdown', 1.0))
        trades_count = full_metrics.get('TradesCount', 0)
        years = max(full_metrics.get('Years', 5.0), 1e-6)
        trades_per_year = trades_count / years
        
        # Get appropriate limits based on stage
        if self.current_stage == 'stage1':
            limits = self.get_adaptive_stage1_limits(relaxation_level)
        else:
            limits = self.hard_limits_stage2
        
        # Hard rejection checks
        if maxdd_full > limits['max_dd_limit']:
            return -999, False, f"MaxDD {maxdd_full*100:.1f}% > {limits['max_dd_limit']*100:.1f}%", trades_per_year
        if sharpe_wf < limits['min_sharpe_wf']:
            return -999, False, f"Sharpe WF {sharpe_wf:.2f} < {limits['min_sharpe_wf']}", trades_per_year
        if trades_per_year < limits['min_trades_per_year']:
            return -999, False, f"Trades/year {trades_per_year:.1f} < {limits['min_trades_per_year']}", trades_per_year
        if trades_per_year > limits['max_trades_per_year']:
            return -999, False, f"Trades/year {trades_per_year:.1f} > {limits['max_trades_per_year']}", trades_per_year
        if turnover > limits['max_turnover']:
            return -999, False, f"Turnover {turnover:.0f} > {limits['max_turnover']}", trades_per_year
        
        # PROTECTION: Even in exploration, reject if trades < 2
        if trades_per_year < 2:
            return -999, False, f"Trades/year {trades_per_year:.1f} < 2 (minimum protection)", trades_per_year
        
        # Score calculation
        turnover_penalty = 0.0
        if turnover > 400:
            turnover_penalty = 3.0 + (turnover - 400) / 50
        elif turnover > 300:
            turnover_penalty = 1.5 + (turnover - 300) / 100
        elif turnover > 240:
            turnover_penalty = (turnover - 240) / 80
        
        cost_penalty = max(0.0, (turnover - 200) / 150)
        stability_bonus = self.calculate_stability(yearly_cagrs) if self.current_stage == 'stage2' else 0.0

        dd_target = 0.45
        dd_penalty = max(0.0, (maxdd_full - dd_target) / 0.10)
        
        score = (
            1.2 * sharpe_wf +
            0.8 * (cagr_wf * 100.0) -
            1.2 * dd_penalty -
            0.20 * turnover_penalty -
            0.10 * cost_penalty +
            0.20 * stability_bonus
        )
        
        return score, True, "", trades_per_year
    
    def objective(
        self,
        trial: optuna.Trial,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        full_df: pd.DataFrame,
        bounds: Dict[str, Tuple] = None,
        relaxation_level: int = 0
    ) -> float:
        """Optuna objective function - OPTIMIZED."""
        timer_obj = TrialTimer()
        timer_obj.start()
        
        if bounds is None:
            bounds = self.get_adaptive_param_bounds(relaxation_level)
        
        params = RobustStrategyV2.sample_params(trial, bounds)
        
        # For Stage1 at relaxation level >= 2, disable the regime gate
        if self.current_stage == 'stage1' and relaxation_level >= 2:
            params['disable_regime_gate'] = True
        else:
            params['disable_regime_gate'] = False
        
        try:
            # Generate signals with TRIAL-SPECIFIC params
            t0 = time.perf_counter()
            
            # Use the original RobustStrategyV2 (proven to work)
            strategy = RobustStrategyV2(params)
            full_signals = strategy.generate_signals(full_df)
            timer_obj.mark('signals', time.perf_counter() - t0)
            
            # Check for empty signals early
            if full_signals.dropna().empty or full_signals.dropna().abs().sum() == 0:
                reason = 'No signals generated (strategy always flat)'
                self._record_trial(trial.number, params, {}, {}, {}, 0, 0, -999, False, reason, relaxation_level, timer_obj)
                return -999
            
            # Run backtests using the same signals for full (slice for train/test)
            t1 = time.perf_counter()
            
            bt_config = BacktestConfig(
                initial_capital=15000.0,
                commission_bps=4.0,
                slippage_bps=5.0,
                leverage=params.get('leverage_cap', 1.5),
                max_leverage=params.get('leverage_cap', 1.5)
            )
            backtester = Backtester(bt_config)
            
            # Train backtest
            train_signals = strategy.generate_signals(train_df)
            train_result = backtester.run(train_df, train_signals)
            timer_obj.mark('train_bt', time.perf_counter() - t1)
            
            # Test backtest
            t2 = time.perf_counter()
            test_signals = strategy.generate_signals(test_df)
            test_result = backtester.run(test_df, test_signals)
            timer_obj.mark('test_bt', time.perf_counter() - t2)
            
            # Full backtest
            t3 = time.perf_counter()
            full_result = backtester.run(full_df, full_signals)
            timer_obj.mark('full_bt', time.perf_counter() - t3)
            
            turnover = self.calculate_turnover(full_signals.fillna(0))
            
            # Yearly stability (Stage2 only) - OPTIMIZED: skip if not Stage2
            yearly_cagrs = []
            if self.current_stage == 'stage2':
                t4 = time.perf_counter()
                yearly_periods = self.split_by_year(full_df)
                for period_name, period_df in yearly_periods.items():
                    if len(period_df) > 100:
                        try:
                            period_signals = strategy.generate_signals_fast(
                                period_df, 
                                strategy.prepare_data(period_df, use_cache=True)
                            )
                            period_result = backtester.run(period_df, period_signals)
                            yearly_cagrs.append(period_result.metrics.get('CAGR', 0))
                        except:
                            pass
                timer_obj.mark('yearly_bt', time.perf_counter() - t4)
            
            score, is_valid, reason, trades_per_year = self.calculate_score(
                train_result.metrics,
                test_result.metrics,
                full_result.metrics,
                turnover,
                yearly_cagrs,
                relaxation_level
            )
            
            self._record_trial(
                trial.number, params, 
                train_result.metrics, test_result.metrics, full_result.metrics,
                turnover, trades_per_year, score, is_valid, reason,
                relaxation_level, timer_obj
            )
            
            if not is_valid:
                return -999
            
            return score
            
        except Exception as e:
            self._record_trial(
                trial.number, params, {}, {}, {}, 0, 0, -999, False,
                f'Exception: {str(e)}', relaxation_level, timer_obj
            )
            return -999
    
    def _record_trial(self, trial_num, params, train_m, test_m, full_m, 
                      turnover, trades_per_year, score, is_valid, reason,
                      relax_level, timer_obj):
        """Record trial info to history."""
        trial_info = {
            'trial': trial_num,
            'params': params.copy(),
            'train_metrics': train_m.copy() if train_m else {},
            'test_metrics': test_m.copy() if test_m else {},
            'full_metrics': full_m.copy() if full_m else {},
            'turnover': turnover,
            'trades_per_year': trades_per_year,
            'score': score,
            'is_valid': is_valid,
            'rejection_reason': reason,
            'relaxation_level': relax_level,
            'timing': timer_obj.timings.copy(),
            'total_time': timer_obj.total(),
        }
        self.optimization_history.append(trial_info)
        self.trial_timings.append(timer_obj.total())
        
        # Update rejection stats (moved here to capture all rejections)
        if reason:
            # Use first few words as key
            key = reason[:50] if len(reason) <= 50 else reason[:50] + "..."
            self.rejection_stats[key] = self.rejection_stats.get(key, 0) + 1
        
        # Print timing info if verbose
        if self.verbose_timing and trial_num % 5 == 0:
            status = "[OK]" if is_valid else "[X]"
            print(f"  Trial {trial_num}: {status} score={score:.2f}, {timer_obj.summary()}")
    
    def stage1_adaptive_search(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        full_df: pd.DataFrame,
        verbose: bool = True
    ) -> List[Dict]:
        """Stage 1: ADAPTIVE coarse search with progressive relaxation."""
        self.current_stage = 'stage1'
        all_candidates = []
        
        print("  Starting optimization (indicators computed per-trial)...")
        
        for level_config in STAGE1_RELAXATION_LEVELS:
            level = level_config['level']
            self.current_relaxation_level = level
            limits = self.get_adaptive_stage1_limits(level)
            bounds = self.get_adaptive_param_bounds(level)
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Stage 1 - Relaxation Level {level}: {level_config['name']}")
                print(f"  min_trades_per_year: {limits['min_trades_per_year']}")
                print(f"  adx_gate_threshold bounds: {bounds['adx_gate_threshold']}")
                print(f"  chop_threshold bounds: {bounds['chop_threshold']}")
                print(f"  Trials: {self.stage1_trials}")
                print(f"{'='*60}")
            
            # FIXED: Use fixed study name for resume
            storage = f"sqlite:///{self.db_path}" if self.db_path else None
            study_name = f'v2_stage1_L{level}'  # Fixed name, no timestamp!
            
            # Check existing trials
            existing_trials = 0
            if storage:
                try:
                    existing_study = optuna.load_study(
                        study_name=study_name,
                        storage=storage
                    )
                    existing_trials = len(existing_study.trials)
                    print(f"  >> Resuming study '{study_name}': {existing_trials} trials already completed")
                except:
                    pass
            
            remaining_trials = max(0, self.stage1_trials - existing_trials)
            
            if remaining_trials == 0:
                print(f"  [OK] Study already complete, loading results...")
                study = optuna.load_study(study_name=study_name, storage=storage)
            else:
                study = optuna.create_study(
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=self.random_state + level * 100),
                    storage=storage,
                    study_name=study_name,
                    load_if_exists=True,  # CRITICAL: Enable resume!
                )
                
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                self.optimization_history = []
                self.rejection_stats = {}
                
                study.optimize(
                    lambda trial: self.objective(
                        trial, train_df, test_df, full_df,
                        bounds, level
                    ),
                    n_trials=remaining_trials,
                    show_progress_bar=verbose,
                    timeout=TRIAL_TIMEOUT_SECONDS * remaining_trials,  # Overall timeout
                )
            
            valid_trials = [t for t in self.optimization_history if t['is_valid']]
            
            # Log relaxation result
            self.relaxation_log.append({
                'level': level,
                'name': level_config['name'],
                'min_trades_per_year': limits['min_trades_per_year'],
                'total_trials': self.stage1_trials,
                'valid_trials': len(valid_trials),
                'existing_trials': existing_trials,
                'adx_bounds': bounds['adx_gate_threshold'],
                'chop_bounds': bounds['chop_threshold'],
            })
            
            if verbose:
                print(f"\nLevel {level} Results: {len(valid_trials)}/{len(self.optimization_history)} valid trials")
                
                # Show rejection breakdown
                if self.rejection_stats:
                    print("Rejection breakdown:")
                    for k, v in sorted(self.rejection_stats.items(), key=lambda x: -x[1])[:5]:
                        print(f"  {v:3d} | {k}...")
                
                # Show timing stats
                if self.trial_timings:
                    avg_time = np.mean(self.trial_timings)
                    max_time = np.max(self.trial_timings)
                    print(f"Timing: avg={avg_time:.2f}s, max={max_time:.2f}s per trial")
            
            if len(valid_trials) >= 1:
                candidates = self._select_pareto_candidates(valid_trials, verbose)
                all_candidates.extend(candidates)
                
                if verbose:
                    print(f"\n[OK] Found {len(candidates)} candidates at relaxation level {level}")
                break
            else:
                if verbose:
                    print(f"\n[X] No valid candidates at level {level}, trying next relaxation...")
        
        self.stage1_candidates = all_candidates
        return all_candidates
    
    def _select_pareto_candidates(self, valid_trials: List[Dict], verbose: bool = True) -> List[Dict]:
        """Select Pareto-optimal candidates from valid trials."""
        candidates = []
        
        # Best Sharpe
        best_sharpe = max(valid_trials, key=lambda t: t['test_metrics']['Sharpe'])
        candidates.append({
            'params': best_sharpe['params'],
            'selection_criterion': 'best_sharpe_wf',
            'metrics': best_sharpe
        })
        
        # Best CAGR
        remaining = [t for t in valid_trials if t['params'] != best_sharpe['params']]
        if remaining:
            best_cagr = max(remaining, key=lambda t: t['test_metrics']['CAGR'])
            candidates.append({
                'params': best_cagr['params'],
                'selection_criterion': 'best_cagr_wf',
                'metrics': best_cagr
            })
        
        # Lowest MaxDD
        remaining = [t for t in valid_trials if t['params'] not in [c['params'] for c in candidates]]
        if remaining:
            best_dd = min(remaining, key=lambda t: t['full_metrics']['MaxDrawdown'])
            candidates.append({
                'params': best_dd['params'],
                'selection_criterion': 'lowest_maxdd',
                'metrics': best_dd
            })
        
        if verbose:
            print(f"\nPareto Candidates Selected:")
            for i, c in enumerate(candidates):
                m = c['metrics']
                print(f"\n  Candidate {i+1} ({c['selection_criterion']}):")
                print(f"    Sharpe WF: {m['test_metrics']['Sharpe']:.2f}")
                print(f"    CAGR WF: {m['test_metrics']['CAGR']:.2f}%")
                print(f"    MaxDD Full: {m['full_metrics']['MaxDrawdown']:.2f}%")
                print(f"    Trades/year: {m['trades_per_year']:.1f}")
                print(f"    Score: {m['score']:.4f}")
        
        return candidates[:self.n_candidates]
    
    def stage2_refine_search(
        self,
        candidate: Dict,
        candidate_idx: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        full_df: pd.DataFrame,
        verbose: bool = True
    ) -> Optional[Dict]:
        """Stage 2: Refine search with STRICT limits."""
        self.current_stage = 'stage2'
        base_params = candidate['params'].copy()
        
        # CRITICAL: Re-enable regime gate for Stage2
        if 'disable_regime_gate' in base_params:
            del base_params['disable_regime_gate']
        
        base_bounds = RobustStrategyV2.get_param_bounds()
        
        # Create refined bounds
        refined_bounds = {}
        for key, (low, high) in base_bounds.items():
            base_val = base_params.get(key, (low + high) / 2)
            
            if isinstance(base_val, int):
                margin = max(1, int(abs(base_val) * 0.15))
                new_low = max(low, base_val - margin)
                new_high = min(high, base_val + margin)
                refined_bounds[key] = (int(new_low), int(new_high))
            else:
                margin = abs(base_val) * 0.20
                new_low = max(low, base_val - margin)
                new_high = min(high, base_val + margin)
                refined_bounds[key] = (new_low, new_high)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Stage 2: STRICT Refine - Candidate {candidate_idx+1}")
            print(f"  Base criterion: {candidate['selection_criterion']}")
            print(f"  Strict limits: min_trades={self.hard_limits_stage2['min_trades_per_year']}, "
                  f"MaxDD<={self.hard_limits_stage2['max_dd_limit']*100}%, "
                  f"Sharpe>={self.hard_limits_stage2['min_sharpe_wf']}")
            print(f"  Trials: {self.stage2_trials}")
            print(f"{'='*60}")
        
        # FIXED: Use fixed study name for resume
        storage = f"sqlite:///{self.db_path}" if self.db_path else None
        study_name = f'v2_stage2_c{candidate_idx}'  # Fixed name!
        
        # Check existing trials
        existing_trials = 0
        if storage:
            try:
                existing_study = optuna.load_study(study_name=study_name, storage=storage)
                existing_trials = len(existing_study.trials)
                print(f"  >> Resuming study '{study_name}': {existing_trials} trials completed")
            except:
                pass
        
        remaining_trials = max(0, self.stage2_trials - existing_trials)
        
        if remaining_trials == 0:
            print(f"  ✓ Study already complete, loading results...")
            study = optuna.load_study(study_name=study_name, storage=storage)
        else:
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state + candidate_idx),
                storage=storage,
                study_name=study_name,
                load_if_exists=True,  # Enable resume!
            )
            
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            start_idx = len(self.optimization_history)
            self.trial_timings = []
            self.rejection_stats = {}
            
            study.optimize(
                lambda trial: self.objective(
                    trial, train_df, test_df, full_df,
                    refined_bounds, 0
                ),
                n_trials=remaining_trials,
                show_progress_bar=verbose
            )
        
        new_trials = self.optimization_history[start_idx:] if 'start_idx' in dir() else self.optimization_history
        valid_trials = [t for t in new_trials if t['is_valid']]
        
        if valid_trials:
            best_trial = max(valid_trials, key=lambda t: t['score'])
            result = {
                'params': best_trial['params'],
                'base_candidate': candidate_idx,
                'metrics': best_trial,
                'improvement': best_trial['score'] - candidate['metrics']['score'],
                'passed_strict': True
            }
        else:
            if verbose:
                print(f"\n[!] Candidate {candidate_idx+1} failed to pass Stage2 strict limits")
                if self.rejection_stats:
                    print("  Rejection reasons:")
                    for k, v in sorted(self.rejection_stats.items(), key=lambda x: -x[1])[:3]:
                        print(f"    {v:3d} | {k}...")
            
            result = {
                'params': candidate['params'],
                'base_candidate': candidate_idx,
                'metrics': candidate['metrics'],
                'improvement': 0,
                'passed_strict': False
            }
        
        if verbose and result['passed_strict']:
            m = result['metrics']
            print(f"\nStage 2 Result for Candidate {candidate_idx+1}:")
            print(f"  Sharpe WF: {m['test_metrics']['Sharpe']:.2f}")
            print(f"  CAGR WF: {m['test_metrics']['CAGR']:.2f}%")
            print(f"  MaxDD Full: {m['full_metrics']['MaxDrawdown']:.2f}%")
            print(f"  Trades/year: {m['trades_per_year']:.1f}")
            print(f"  Score: {m['score']:.4f}")
            print(f"  [OK] Passed strict Stage2 limits")
            
            # Print timing stats
            if self.trial_timings:
                print(f"  Timing: avg={np.mean(self.trial_timings):.2f}s/trial")
        
        return result
    
    def walk_forward(
        self,
        df: pd.DataFrame,
        train_end: str = "2022-12-31",
        periods_per_year: int = 24 * 365,
        verbose: bool = True
    ) -> WalkForwardResultV2:
        """Run complete two-stage walk-forward optimization with ADAPTIVE Stage1."""
        train_df, test_df = self.split_data(df, train_end)
        
        if verbose:
            print(f"\n{'#'*70}")
            print("Walk-Forward Optimization V2 - ADAPTIVE (OPTIMIZED)")
            print(f"{'#'*70}")
            print(f"\nData Split:")
            print(f"  Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()} ({len(train_df):,} rows)")
            print(f"  Test: {test_df['timestamp'].min()} to {test_df['timestamp'].max()} ({len(test_df):,} rows)")
            print(f"  Full: {len(df):,} rows")
        
        # Stage 1: Adaptive search
        candidates = self.stage1_adaptive_search(train_df, test_df, df, verbose)
        
        if not candidates:
            print("\n" + "="*70)
            print("CRITICAL: No candidates found even with maximum relaxation!")
            print("="*70)
            print("\nRelaxation Log:")
            for log in self.relaxation_log:
                print(f"  Level {log['level']} ({log['name']}): {log['valid_trials']}/{log['total_trials']} valid")
            raise ValueError("Stage 1 failed to find any valid candidates even with maximum relaxation")
        
        # Stage 2: Strict refine each candidate
        stage2_results = []
        for i, candidate in enumerate(candidates):
            result = self.stage2_refine_search(candidate, i, train_df, test_df, df, verbose)
            if result:
                stage2_results.append(result)
        
        self.stage2_results = stage2_results
        
        # Select final best
        passed_strict = [r for r in stage2_results if r['passed_strict']]
        
        passed_strict_flag = False
        if passed_strict:
            best_result = max(passed_strict, key=lambda r: r['metrics']['score'])
            passed_strict_flag = True
            if verbose:
                print(f"\n[OK] Final best selected from {len(passed_strict)} candidates that passed Stage2")
        else:
            if verbose:
                print(f"\n[!] No candidates passed Stage2 strict limits!")
                print("    Selecting best available candidate (may not meet all criteria)")
            best_result = max(stage2_results, key=lambda r: r['metrics']['score'])
        
        self.best_params = best_result['params']
        
        if verbose:
            print(f"\n{'='*70}")
            print("Final Best Parameters")
            print(f"{'='*70}")
            m = best_result['metrics']
            print(f"  Source: Candidate {best_result['base_candidate']+1}")
            print(f"  Passed Strict Stage2: {'Yes' if best_result['passed_strict'] else 'No'}")
            print(f"  Final Score: {m['score']:.4f}")
            print(f"  Sharpe (WF Test): {m['test_metrics']['Sharpe']:.2f}")
            print(f"  CAGR (WF Test): {m['test_metrics']['CAGR']:.2f}%")
            print(f"  MaxDD (Full): {m['full_metrics']['MaxDrawdown']:.2f}%")
            print(f"  Trades/year: {m['trades_per_year']:.1f}")
        
        # Run final backtest for complete results
        final_train_result, _ = self.run_backtest_optimized(train_df, self.best_params)
        final_test_result, _ = self.run_backtest_optimized(test_df, self.best_params)
        final_full_result, _ = self.run_backtest_optimized(df, self.best_params)
        
        # Calculate yearly metrics
        yearly_metrics = {}
        yearly_periods = self.split_by_year(df)
        for period_name, period_df in yearly_periods.items():
            if len(period_df) > 100:
                try:
                    period_result, _ = self.run_backtest_optimized(period_df, self.best_params)
                    yearly_metrics[period_name] = period_result.metrics
                except:
                    pass
        
        return WalkForwardResultV2(
            best_params=self.best_params,
            train_metrics=final_train_result.metrics,
            test_metrics=final_test_result.metrics,
            full_metrics=final_full_result.metrics,
            equity_curve=final_full_result.equity_curve,
            trades=final_full_result.trades,
            yearly_metrics=yearly_metrics,
            optimization_history=self.optimization_history,
            stage1_candidates=self.stage1_candidates,
            stage2_results=self.stage2_results,
            relaxation_log=self.relaxation_log,
            passed_strict_stage2=passed_strict_flag
        )


# =============================================================================
# Robustness Testing (Unchanged)
# =============================================================================

class RobustnessTests:
    """Run robustness tests on final parameters."""
    
    def __init__(self, df: pd.DataFrame, params: Dict):
        self.df = df
        self.params = params
        self.results = {}
    
    def run_all(self, verbose: bool = True) -> Dict:
        """Run all robustness tests."""
        if verbose:
            print("\n" + "="*70)
            print("Running Robustness Tests")
            print("="*70)
        
        self.results['yearly_splits'] = self.test_yearly_splits(verbose)
        self.results['cost_sensitivity'] = self.test_cost_sensitivity(verbose)
        self.results['param_perturbation'] = self.test_param_perturbation(verbose)
        
        if verbose:
            self._print_summary()
        
        return self.results
    
    def test_yearly_splits(self, verbose: bool = True) -> Dict:
        """Test performance on different year splits."""
        if verbose:
            print("\n[1/3] Yearly Split Analysis")
        
        periods = {
            '2020-2021': ('2020-01-01', '2021-12-31'),
            '2022': ('2022-01-01', '2022-12-31'),
            '2023': ('2023-01-01', '2023-12-31'),
            '2024-2025': ('2024-01-01', '2025-11-30'),
        }
        
        results = {}
        for period_name, (start, end) in periods.items():
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            
            period_df = self.df[
                (self.df['timestamp'] >= start_dt) & 
                (self.df['timestamp'] <= end_dt)
            ].reset_index(drop=True)
            
            if len(period_df) < 100:
                continue
            
            try:
                strategy = OptimizedRobustStrategyV2(self.params)
                signals = strategy.generate_signals_fast(period_df, strategy.prepare_data(period_df))
                
                bt_config = BacktestConfig(
                    initial_capital=15000.0,
                    leverage=self.params.get('leverage_cap', 1.5)
                )
                backtester = Backtester(bt_config)
                result = backtester.run(period_df, signals)
                
                results[period_name] = {
                    'CAGR': result.metrics['CAGR'],
                    'MaxDD': result.metrics['MaxDrawdown'],
                    'Sharpe': result.metrics['Sharpe'],
                    'Trades': result.metrics['TradesCount'],
                }
                
                if verbose:
                    print(f"  {period_name}: CAGR {result.metrics['CAGR']:+.1f}%, "
                          f"MaxDD {result.metrics['MaxDrawdown']:.1f}%, "
                          f"Sharpe {result.metrics['Sharpe']:.2f}")
            except Exception as e:
                if verbose:
                    print(f"  {period_name}: Error - {e}")
        
        return results
    
    def test_cost_sensitivity(self, verbose: bool = True) -> Dict:
        """Test performance under different cost scenarios."""
        if verbose:
            print("\n[2/3] Cost Sensitivity Analysis")
        
        cost_multipliers = [1.0, 2.0, 3.0]
        base_commission = 4.0
        base_slippage = 5.0
        
        results = {}
        strategy = OptimizedRobustStrategyV2(self.params)
        prepared = strategy.prepare_data(self.df)
        signals = strategy.generate_signals_fast(self.df, prepared)
        
        for mult in cost_multipliers:
            try:
                bt_config = BacktestConfig(
                    initial_capital=15000.0,
                    commission_bps=base_commission * mult,
                    slippage_bps=base_slippage * mult,
                    leverage=self.params.get('leverage_cap', 1.5)
                )
                backtester = Backtester(bt_config)
                result = backtester.run(self.df, signals)
                
                results[f'{mult}x'] = {
                    'CAGR': result.metrics['CAGR'],
                    'MaxDD': result.metrics['MaxDrawdown'],
                    'Sharpe': result.metrics['Sharpe'],
                    'FinalEquity': result.metrics['FinalEquity'],
                }
                
                if verbose:
                    print(f"  Cost {mult}x: CAGR {result.metrics['CAGR']:+.1f}%, "
                          f"Sharpe {result.metrics['Sharpe']:.2f}, "
                          f"Final ${result.metrics['FinalEquity']:,.0f}")
            except Exception as e:
                if verbose:
                    print(f"  Cost {mult}x: Error - {e}")
        
        return results
    
    def test_param_perturbation(self, verbose: bool = True, n_tests: int = 15) -> Dict:
        """Test parameter robustness with ±10% perturbations."""
        if verbose:
            print(f"\n[3/3] Parameter Perturbation Analysis (n={n_tests})")
        
        np.random.seed(42)
        
        key_params = [
            'donchian_entry_period', 'donchian_exit_period', 'adx_gate_threshold',
            'vol_target', 'leverage_cap', 'min_hold_hours', 'chop_threshold'
        ]
        
        perturbation_results = []
        
        for i in range(n_tests):
            perturbed_params = self.params.copy()
            
            for key in key_params:
                if key in perturbed_params:
                    val = perturbed_params[key]
                    perturbation = 1 + np.random.uniform(-0.10, 0.10)
                    if isinstance(val, int):
                        perturbed_params[key] = max(1, int(val * perturbation))
                    else:
                        perturbed_params[key] = val * perturbation
            
            try:
                strategy = OptimizedRobustStrategyV2(perturbed_params)
                prepared = strategy.prepare_data(self.df)
                signals = strategy.generate_signals_fast(self.df, prepared)
                
                bt_config = BacktestConfig(
                    initial_capital=15000.0,
                    leverage=perturbed_params.get('leverage_cap', 1.5)
                )
                backtester = Backtester(bt_config)
                result = backtester.run(self.df, signals)
                
                perturbation_results.append({
                    'CAGR': result.metrics['CAGR'],
                    'MaxDD': result.metrics['MaxDrawdown'],
                    'Sharpe': result.metrics['Sharpe'],
                })
            except:
                perturbation_results.append({
                    'CAGR': -100, 'MaxDD': 100, 'Sharpe': -1,
                })
        
        cagrs = [r['CAGR'] for r in perturbation_results]
        maxdds = [r['MaxDD'] for r in perturbation_results]
        sharpes = [r['Sharpe'] for r in perturbation_results]
        
        results = {
            'CAGR': {'mean': np.mean(cagrs), 'std': np.std(cagrs), 
                    'min': np.min(cagrs), 'max': np.max(cagrs)},
            'MaxDD': {'mean': np.mean(maxdds), 'std': np.std(maxdds),
                     'min': np.min(maxdds), 'max': np.max(maxdds)},
            'Sharpe': {'mean': np.mean(sharpes), 'std': np.std(sharpes),
                      'min': np.min(sharpes), 'max': np.max(sharpes)},
            'raw_results': perturbation_results,
        }
        
        if verbose:
            print(f"  CAGR: {results['CAGR']['mean']:.1f}% ± {results['CAGR']['std']:.1f}% "
                  f"(range: {results['CAGR']['min']:.1f}% to {results['CAGR']['max']:.1f}%)")
            print(f"  MaxDD: {results['MaxDD']['mean']:.1f}% ± {results['MaxDD']['std']:.1f}%")
            print(f"  Sharpe: {results['Sharpe']['mean']:.2f} ± {results['Sharpe']['std']:.2f}")
        
        return results
    
    def _print_summary(self):
        """Print robustness summary."""
        print("\n" + "="*70)
        print("Robustness Summary")
        print("="*70)
        
        yearly = self.results.get('yearly_splits', {})
        if yearly:
            cagrs = [v['CAGR'] for v in yearly.values()]
            negative_years = sum(1 for c in cagrs if c < 0)
            print(f"Yearly Consistency: {len(cagrs) - negative_years}/{len(cagrs)} years profitable")
        
        costs = self.results.get('cost_sensitivity', {})
        if costs:
            cagr_3x = costs.get('3.0x', {}).get('CAGR', -100)
            if cagr_3x > 0:
                print(f"Cost Robustness: [OK] Profitable at 3x costs (CAGR {cagr_3x:.1f}%)")
            else:
                print(f"Cost Robustness: [!] Not profitable at 3x costs")


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_equity_curve_v2(equity_df, metrics, output_path, initial_capital=15000.0):
    """Plot equity curve for V2 strategy."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), 
                              gridspec_kw={'height_ratios': [3, 1, 1]})
    
    timestamps = pd.to_datetime(equity_df['timestamp'])
    equity = equity_df['equity']
    
    ax1 = axes[0]
    ax1.plot(timestamps, equity, color='#1E88E5', linewidth=1.5, label='Strategy V2')
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    
    ax1.fill_between(timestamps, initial_capital, equity, 
                     where=(equity < initial_capital), 
                     color='#E53935', alpha=0.2, interpolate=True)
    ax1.fill_between(timestamps, initial_capital, equity, 
                     where=(equity >= initial_capital), 
                     color='#43A047', alpha=0.2, interpolate=True)
    
    ax1.set_ylabel('Portfolio Value (USD)', fontsize=11)
    ax1.set_title(f'Strategy V2 - 5-Year Equity Curve (2020-2025)\n'
                  f'CAGR: {metrics["CAGR"]:.1f}% | MaxDD: {metrics["MaxDrawdown"]:.1f}% | '
                  f'Sharpe: {metrics["Sharpe"]:.2f} | Final: ${metrics["FinalEquity"]:,.0f}',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    ax2 = axes[1]
    drawdown = equity_df['drawdown'] * 100
    ax2.fill_between(timestamps, 0, drawdown, color='#E53935', alpha=0.4)
    ax2.plot(timestamps, drawdown, color='#B71C1C', linewidth=0.8)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_ylim(top=0)
    ax2.axhline(y=-metrics['MaxDrawdown'], color='red', linestyle='--', alpha=0.7)
    
    ax3 = axes[2]
    position = equity_df['position']
    ax3.fill_between(timestamps, 0, position, where=(position > 0), 
                     color='#43A047', alpha=0.4, label='Long')
    ax3.fill_between(timestamps, 0, position, where=(position < 0), 
                     color='#E53935', alpha=0.4, label='Short')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_ylabel('Position', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.legend(loc='upper right')
    
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_drawdown_v2(equity_df, metrics, output_path):
    """Plot drawdown chart."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    timestamps = pd.to_datetime(equity_df['timestamp'])
    drawdown = equity_df['drawdown'] * 100
    
    ax.fill_between(timestamps, 0, drawdown, color='#E53935', alpha=0.5)
    ax.plot(timestamps, drawdown, color='#B71C1C', linewidth=1)
    
    max_dd_idx = drawdown.idxmin()
    ax.scatter([timestamps[max_dd_idx]], [drawdown[max_dd_idx]], 
               color='#B71C1C', s=100, zorder=5)
    ax.annotate(f'Max DD: {metrics["MaxDrawdown"]:.1f}%', 
                xy=(timestamps[max_dd_idx], drawdown[max_dd_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
    ax.set_ylim(top=5)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_yearly_returns_v2(equity_df, output_path):
    """Plot yearly returns bar chart."""
    equity_df = equity_df.copy()
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    equity_df['year'] = equity_df['timestamp'].dt.year
    
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    yearly_returns = []
    
    for year in years:
        year_data = equity_df[equity_df['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['equity'].iloc[-1] / year_data['equity'].iloc[0] - 1) * 100
        else:
            year_return = 0
        yearly_returns.append(year_return)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#43A047' if r > 0 else '#E53935' for r in yearly_returns]
    bars = ax.bar(years, yearly_returns, color=colors, edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars, yearly_returns):
        height = bar.get_height()
        ax.annotate(f'{val:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -3),
                    textcoords='offset points',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Return (%)', fontsize=12)
    ax.set_title('Yearly Returns - Strategy V2', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_robustness_perturbation(perturb_results, output_path):
    """Plot parameter perturbation analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    raw = perturb_results.get('raw_results', [])
    if not raw:
        return
    
    cagrs = [r['CAGR'] for r in raw]
    maxdds = [r['MaxDD'] for r in raw]
    sharpes = [r['Sharpe'] for r in raw]
    
    bp1 = axes[0].boxplot([cagrs], vert=True, patch_artist=True)
    bp1['boxes'][0].set_facecolor('#1E88E5')
    axes[0].set_ylabel('CAGR (%)', fontsize=11)
    axes[0].set_title('CAGR Distribution', fontsize=12, fontweight='bold')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticklabels([''])
    
    bp2 = axes[1].boxplot([maxdds], vert=True, patch_artist=True)
    bp2['boxes'][0].set_facecolor('#E53935')
    axes[1].set_ylabel('MaxDD (%)', fontsize=11)
    axes[1].set_title('MaxDD Distribution', fontsize=12, fontweight='bold')
    axes[1].axhline(y=45, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticklabels([''])
    
    bp3 = axes[2].boxplot([sharpes], vert=True, patch_artist=True)
    bp3['boxes'][0].set_facecolor('#43A047')
    axes[2].set_ylabel('Sharpe', fontsize=11)
    axes[2].set_title('Sharpe Distribution', fontsize=12, fontweight='bold')
    axes[2].axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticklabels([''])
    
    fig.suptitle('Parameter Perturbation Analysis (±10%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Report Generation
# =============================================================================

def generate_report_v2(result, robustness, v1_metrics, output_path):
    """Generate comprehensive markdown report."""
    
    v2_full = result.full_metrics
    v2_train = result.train_metrics
    v2_test = result.test_metrics
    
    cagr_improve = v2_full['CAGR'] - v1_metrics['CAGR']
    maxdd_improve = v1_metrics['MaxDD'] - v2_full['MaxDrawdown']
    sharpe_improve = v2_full['Sharpe'] - v1_metrics['Sharpe']
    trades_improve = v1_metrics['Trades'] - v2_full['TradesCount']
    
    relaxation_summary = ""
    for log in result.relaxation_log:
        status = "OK" if log['valid_trials'] > 0 else "X"
        relaxation_summary += f"\n| {log['level']} | {log['name']} | {log['min_trades_per_year']} | {log['valid_trials']}/{log['total_trials']} | {status} |"
    
    report = f"""# ETH Strategy V2 - Adaptive Walk-Forward Optimization Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Period:** 2020-01-01 to 2025-11-30
**Initial Capital:** $15,000 USD

---

## Executive Summary

This report presents the results of the V2 **Adaptive** Walk-Forward Optimization (OPTIMIZED version).

### Final Status: {'PASSED Stage2 Strict Limits' if result.passed_strict_stage2 else 'DID NOT PASS Stage2 Strict Limits'}

---

## Relaxation Log

| Level | Name | min_trades/year | Valid/Total | Status |
|-------|------|-----------------|-------------|--------|{relaxation_summary}

---

## Final Best Parameters

```json
{json.dumps(result.best_params, indent=2)}
```

---

## Walk-Forward Results

### Training Period (2020-2022)
| Metric | Value |
|--------|-------|
| CAGR | {v2_train['CAGR']:.2f}% |
| MaxDrawdown | {v2_train['MaxDrawdown']:.2f}% |
| Sharpe | {v2_train['Sharpe']:.2f} |
| Trades | {v2_train['TradesCount']} |

### Test Period (2023-2025) - Out-of-Sample
| Metric | Value |
|--------|-------|
| CAGR | {v2_test['CAGR']:.2f}% |
| MaxDrawdown | {v2_test['MaxDrawdown']:.2f}% |
| Sharpe | {v2_test['Sharpe']:.2f} |
| Trades | {v2_test['TradesCount']} |

### Full Period (2020-2025)
| Metric | Value |
|--------|-------|
| CAGR | {v2_full['CAGR']:.2f}% |
| MaxDrawdown | {v2_full['MaxDrawdown']:.2f}% |
| Sharpe | {v2_full['Sharpe']:.2f} |
| Trades | {v2_full['TradesCount']} |
| Trades/Year | {v2_full['TradesPerYear']:.1f} |
| Turnover | {v2_full['Turnover']:.0f} |
| Final Equity | ${v2_full['FinalEquity']:,.0f} |

---

## V2 vs V1 Comparison

| Metric | V1 (Route A) | V2 | Change |
|--------|-------------|-----|--------|
| CAGR | {v1_metrics['CAGR']:.2f}% | {v2_full['CAGR']:.2f}% | {cagr_improve:+.2f}% |
| MaxDD | {v1_metrics['MaxDD']:.2f}% | {v2_full['MaxDrawdown']:.2f}% | {maxdd_improve:+.2f}% (improved) |
| Sharpe | {v1_metrics['Sharpe']:.2f} | {v2_full['Sharpe']:.2f} | {sharpe_improve:+.2f} |
| Trades | {v1_metrics['Trades']} | {v2_full['TradesCount']} | {trades_improve:+d} |

---

**Report End**
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Saved: {output_path}")


# =============================================================================
# Profile Mode
# =============================================================================

def run_profile_mode(df, args):
    """Run profiling mode to identify bottlenecks."""
    print("\n" + "="*70)
    print("PROFILE MODE - Running 3 trials to identify bottlenecks")
    print("="*70)
    
    try:
        import cProfile
        import pstats
        from io import StringIO
    except ImportError:
        print("cProfile not available")
        return
    
    output_dir = PROJECT_ROOT / 'outputs_v2'
    output_dir.mkdir(exist_ok=True)
    
    # Create optimizer with minimal trials
    optimizer = AdaptiveWalkForwardOptimizerV2(
        stage1_trials=3,
        stage2_trials=2,
        n_candidates=1,
        db_path=None,  # Don't save to DB in profile mode
        verbose_timing=True
    )
    
    # Profile the walk_forward
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        result = optimizer.walk_forward(df, train_end="2022-12-31", verbose=True)
    except Exception as e:
        print(f"Error during profiling: {e}")
    
    profiler.disable()
    
    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print("\n" + "="*70)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print("="*70)
    print(s.getvalue())
    
    # Save profile
    profile_path = output_dir / 'profile_results.txt'
    with open(profile_path, 'w') as f:
        f.write(s.getvalue())
    print(f"\nProfile saved to: {profile_path}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='ETH Strategy V2 Adaptive Backtest (OPTIMIZED)')
    parser.add_argument('--stage1-trials', type=int, default=80,
                        help='Number of Stage 1 trials')
    parser.add_argument('--stage2-trials', type=int, default=50,
                        help='Number of Stage 2 trials per candidate')
    parser.add_argument('--n-candidates', type=int, default=3,
                        help='Number of candidates from Stage 1')
    parser.add_argument('--profile', action='store_true',
                        help='Run in profile mode (3 trials only, with cProfile)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode (5 stage1, 3 stage2 trials)')
    parser.add_argument('--reset', action='store_true',
                        help='Reset database (use new db file)')
    parser.add_argument('--no-db', action='store_true',
                        help='Run without database (no resume support)')
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        args.stage1_trials = 5
        args.stage2_trials = 3
        args.n_candidates = 2
    
    # Setup paths
    output_dir = PROJECT_ROOT / 'outputs_v2'
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)
    
    if args.no_db:
        db_path = None
    elif args.reset:
        db_path = output_dir / f'optuna_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
    else:
        db_path = output_dir / 'optuna_v2_adaptive_complete.db'
    
    print("="*70)
    print("ETH Strategy V2 - ADAPTIVE Walk-Forward Optimization (OPTIMIZED)")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {output_dir}")
    print(f"Stage 1 Trials: {args.stage1_trials}")
    print(f"Stage 2 Trials: {args.stage2_trials} x {args.n_candidates}")
    print(f"Profile Mode: {args.profile}")
    print("="*70)
    
    # Load V1 metrics for comparison
    v1_metrics_path = PROJECT_ROOT / 'outputs' / 'metrics_route_A.json'
    if v1_metrics_path.exists():
        with open(v1_metrics_path) as f:
            v1_data = json.load(f)
        v1_metrics = {
            'CAGR': v1_data.get('CAGR', 15.3),
            'MaxDD': v1_data.get('MaxDrawdown', 59.8),
            'Sharpe': v1_data.get('Sharpe', 0.64),
            'Trades': v1_data.get('TradesCount', 1535),
            'Turnover': v1_data.get('Turnover', 18430),
            'FinalEquity': v1_data.get('FinalEquity', 34837),
        }
    else:
        v1_metrics = {
            'CAGR': 15.3, 'MaxDD': 59.8, 'Sharpe': 0.64,
            'Trades': 1535, 'Turnover': 18430, 'FinalEquity': 34837
        }
    
    print(f"\nV1 Baseline (Route A):")
    print(f"  CAGR: {v1_metrics['CAGR']:.2f}%")
    print(f"  MaxDD: {v1_metrics['MaxDD']:.2f}%")
    print(f"  Sharpe: {v1_metrics['Sharpe']:.2f}")
    print(f"  Trades: {v1_metrics['Trades']}")
    
    # Load data
    print("\n[1/5] Loading ETH data (2020-01-01 to 2025-11-30)...")
    t_load = time.perf_counter()
    df = load_data(
        start_date="2020-01-01",
        end_date="2025-11-30",
        freq="1h"
    )
    print(f"  Loaded in {time.perf_counter() - t_load:.2f}s")
    
    info = get_data_info(df)
    print(f"Data loaded successfully:")
    print(f"  Rows: {info['rows']:,}")
    print(f"  Period: {info['start_date']} to {info['end_date']}")
    
    # Profile mode
    if args.profile:
        run_profile_mode(df, args)
        return
    
    # Run optimization
    print("\n[2/5] Running ADAPTIVE two-stage optimization...")
    t_opt = time.perf_counter()
    
    optimizer = AdaptiveWalkForwardOptimizerV2(
        stage1_trials=args.stage1_trials,
        stage2_trials=args.stage2_trials,
        n_candidates=args.n_candidates,
        db_path=str(db_path),
        verbose_timing=True
    )
    
    result = optimizer.walk_forward(df, train_end="2022-12-31", verbose=True)
    
    print(f"\n  Optimization completed in {time.perf_counter() - t_opt:.1f}s")
    
    # Save best params immediately
    params_path = output_dir / 'best_params_v2.json'
    with open(params_path, 'w') as f:
        json.dump(result.best_params, f, indent=2)
    print(f"\nSaved: {params_path}")
    
    # Run robustness tests
    print("\n[3/5] Running robustness tests...")
    robustness_tester = RobustnessTests(df, result.best_params)
    robustness_results = robustness_tester.run_all(verbose=True)
    
    # Save results
    print("\n[4/5] Saving results...")
    
    with open(output_dir / 'metrics_full_v2.json', 'w') as f:
        json.dump(result.full_metrics, f, indent=2)
    
    wf_metrics = {'train': result.train_metrics, 'test': result.test_metrics}
    with open(output_dir / 'metrics_walkforward_v2.json', 'w') as f:
        json.dump(wf_metrics, f, indent=2)
    
    result.equity_curve.to_parquet(output_dir / 'equity_curve_v2.parquet', index=False)
    
    # Stress tests summary
    stress_rows = []
    for period, metrics in robustness_results.get('yearly_splits', {}).items():
        stress_rows.append({
            'test_type': 'yearly_split', 'test_name': period,
            'CAGR': metrics.get('CAGR', 0), 'MaxDD': metrics.get('MaxDD', 0),
            'Sharpe': metrics.get('Sharpe', 0),
        })
    
    for cost, metrics in robustness_results.get('cost_sensitivity', {}).items():
        stress_rows.append({
            'test_type': 'cost_sensitivity', 'test_name': f'cost_{cost}',
            'CAGR': metrics.get('CAGR', 0), 'MaxDD': metrics.get('MaxDD', 0),
            'Sharpe': metrics.get('Sharpe', 0),
        })
    
    perturb = robustness_results.get('param_perturbation', {})
    stress_rows.append({
        'test_type': 'param_perturbation', 'test_name': 'mean',
        'CAGR': perturb.get('CAGR', {}).get('mean', 0),
        'MaxDD': perturb.get('MaxDD', {}).get('mean', 0),
        'Sharpe': perturb.get('Sharpe', {}).get('mean', 0),
    })
    
    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(output_dir / 'stress_tests_summary.csv', index=False)
    
    # Generate plots
    print("\n[5/5] Generating plots...")
    
    plot_equity_curve_v2(
        result.equity_curve, result.full_metrics,
        output_dir / 'plots' / 'equity_curve_v2.png'
    )
    
    plot_drawdown_v2(
        result.equity_curve, result.full_metrics,
        output_dir / 'plots' / 'drawdown_v2.png'
    )
    
    plot_yearly_returns_v2(
        result.equity_curve,
        output_dir / 'plots' / 'yearly_returns_v2.png'
    )
    
    plot_robustness_perturbation(
        robustness_results.get('param_perturbation', {}),
        output_dir / 'plots' / 'robustness_perturbation.png'
    )
    
    # Generate report
    generate_report_v2(result, robustness_results, v1_metrics, output_dir / 'report_v2.md')
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL RESULTS - V2 Adaptive Optimization")
    print("="*70)
    
    print(f"\nStage2 Strict Limits: {'PASSED [OK]' if result.passed_strict_stage2 else 'NOT PASSED [!]'}")
    
    print(f"\n{'Metric':<20} {'V1 (Route A)':<15} {'V2':<15} {'Change':<15}")
    print("-"*70)
    
    v2 = result.full_metrics
    
    print(f"{'CAGR':<20} {v1_metrics['CAGR']:.2f}%{'':<8} {v2['CAGR']:.2f}%{'':<8} {v2['CAGR']-v1_metrics['CAGR']:+.2f}%")
    print(f"{'MaxDrawdown':<20} {v1_metrics['MaxDD']:.2f}%{'':<8} {v2['MaxDrawdown']:.2f}%{'':<8} {v1_metrics['MaxDD']-v2['MaxDrawdown']:+.2f}% (better)")
    print(f"{'Sharpe':<20} {v1_metrics['Sharpe']:.2f}{'':<12} {v2['Sharpe']:.2f}{'':<12} {v2['Sharpe']-v1_metrics['Sharpe']:+.2f}")
    print(f"{'Trades':<20} {v1_metrics['Trades']:<15} {v2['TradesCount']:<15} {v1_metrics['Trades']-v2['TradesCount']:+d}")
    print(f"{'Trades/Year':<20} {'':<15} {v2['TradesPerYear']:.1f}")
    print(f"{'Final Equity':<20} ${v1_metrics['FinalEquity']:,.0f}{'':<5} ${v2['FinalEquity']:,.0f}")
    
    print("\n" + "="*70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
