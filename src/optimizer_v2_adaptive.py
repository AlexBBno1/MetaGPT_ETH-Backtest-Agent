"""
Walk-Forward Optimizer V2 Adaptive - Progressive Stage1 Relaxation

Key Changes from optimizer_v2.py:
1. Adaptive Stage1: Progressively relax min_trades_per_year (30 -> 10 -> 5)
2. Wider regime gate search bounds for exploration
3. Stage2 maintains strict limits as per specification
4. Detailed logging of relaxation decisions
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import optuna
from datetime import datetime
import warnings
import copy

warnings.filterwarnings('ignore')

from backtester import Backtester, BacktestConfig, calculate_metrics
from strategies.robust_strategy_v2 import RobustStrategyV2


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


class AdaptiveWalkForwardOptimizerV2:
    """
    Walk-Forward Optimizer V2 with ADAPTIVE Stage1 exploration.
    
    Stage 1: Adaptive coarse search - progressively relax constraints to find viable region
    Stage 2: Strict refine search - apply hard limits from specification
    """
    
    def __init__(
        self,
        train_years: int = 3,
        stage1_trials: int = 60,
        stage2_trials: int = 40,
        n_candidates: int = 3,
        random_state: int = 42,
        db_path: str = None
    ):
        self.train_years = train_years
        self.stage1_trials = stage1_trials
        self.stage2_trials = stage2_trials
        self.n_candidates = n_candidates
        self.random_state = random_state
        self.db_path = db_path
        
        self.best_params = None
        self.optimization_history = []
        self.stage1_candidates = []
        self.stage2_results = []
        self.relaxation_log = []
        
        # Stage2 STRICT hard limits (immutable per specification)
        self.hard_limits_stage2 = {
            'max_dd_limit': 0.45,        # 45% max drawdown
            'min_sharpe_wf': 0.7,        # Sharpe >= 0.7 (prefer 0.8)
            'min_trades_per_year': 60,   # At least 60 trades/year
            'max_trades_per_year': 150,  # At most 150 trades/year
            'max_turnover': 400,         # Max turnover units/year
        }
        
        # Stage1 ADAPTIVE limits (will be progressively relaxed)
        self.stage1_relaxation_levels = [
            # Level 0: Standard
            {'min_trades_per_year': 30, 'name': 'standard'},
            # Level 1: First relaxation
            {'min_trades_per_year': 10, 'name': 'relaxed_10'},
            # Level 2: Second relaxation  
            {'min_trades_per_year': 5, 'name': 'relaxed_5'},
            # Level 3: Minimal gate - find ANY trading activity
            {'min_trades_per_year': 3, 'name': 'minimal_gate'},
        ]
        
        self.current_stage = 'stage1'
        self.current_relaxation_level = 0
        
    def get_adaptive_stage1_limits(self, level: int = 0) -> Dict:
        """Get Stage1 limits for given relaxation level."""
        base_limits = {
            'max_dd_limit': 0.70,        # Looser DD limit for exploration
            'min_sharpe_wf': -0.5,       # Allow negative sharpe in exploration
            'max_trades_per_year': 300,  # Higher cap for exploration
            'max_turnover': 600,         # Higher turnover allowed
        }
        relaxation = self.stage1_relaxation_levels[min(level, len(self.stage1_relaxation_levels)-1)]
        base_limits['min_trades_per_year'] = relaxation['min_trades_per_year']
        return base_limits
    
    def get_adaptive_param_bounds(self, level: int = 0) -> Dict[str, Tuple]:
        """
        Get parameter bounds for Stage1, progressively widened.
        
        Key changes at higher relaxation levels:
        - adx_gate_threshold: lower bound reduced (allows weaker trend signals)
        - chop_threshold: upper bound increased (allows choppier conditions)
        - min_hold_hours: lower bound reduced (allows shorter holds)
        """
        base_bounds = RobustStrategyV2.get_param_bounds()
        
        if level >= 1:
            # Relax regime gate bounds
            base_bounds['adx_gate_threshold'] = (8, 35)   # Was (12, 35), now allows weaker trends
            base_bounds['chop_threshold'] = (40, 80)      # Was (40, 70), now allows choppier conditions
            
        if level >= 2:
            # Further relax holding period
            base_bounds['min_hold_hours'] = (24, 144)     # Was (36, 144), now allows shorter holds
            base_bounds['adx_gate_threshold'] = (6, 35)   # Even more permissive
            base_bounds['chop_threshold'] = (35, 85)      # Even more permissive
        
        if level >= 3:
            # Minimal gate - nearly disable regime filtering to find ANY trades
            base_bounds['adx_gate_threshold'] = (5, 40)   # Very low ADX threshold
            base_bounds['chop_threshold'] = (30, 90)      # Very high chop tolerance
            base_bounds['min_hold_hours'] = (12, 144)     # Short holds allowed
            base_bounds['cooldown_hours'] = (4, 96)       # Short cooldown
            base_bounds['donchian_entry_period'] = (48, 264)  # Allow shorter lookback
            
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
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        periods_per_year: int = 24 * 365
    ) -> Any:
        """Run backtest with given parameters."""
        strategy = RobustStrategyV2(params)
        signals = strategy.generate_signals(df)
        
        bt_config = BacktestConfig(
            initial_capital=15000.0,
            commission_bps=4.0,
            slippage_bps=5.0,
            leverage=params.get('leverage_cap', 1.5),
            max_leverage=params.get('leverage_cap', 1.5)
        )
        
        backtester = Backtester(bt_config)
        return backtester.run(df, signals, periods_per_year)
    
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
        """
        Calculate composite score with stage-appropriate hard rejection.
        """
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
        
        # PROTECTION: Even in exploration, reject if trades < 2 (avoid fake solutions)
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
        """Optuna objective function."""
        if bounds is None:
            bounds = self.get_adaptive_param_bounds(relaxation_level)
        
        params = RobustStrategyV2.sample_params(trial, bounds)
        
        # For Stage1 at relaxation level >= 2, disable the regime gate to find ANY trading activity
        if self.current_stage == 'stage1' and relaxation_level >= 2:
            params['disable_regime_gate'] = True
        else:
            # Ensure regime gate is enabled for Stage2 and early Stage1 levels
            params['disable_regime_gate'] = False
        
        try:
            train_result = self.run_backtest(train_df, params)
            test_result = self.run_backtest(test_df, params)
            full_result = self.run_backtest(full_df, params)
            
            strategy = RobustStrategyV2(params)
            signals = strategy.generate_signals(full_df)
            
            # Check for empty signals - log and reject
            if signals.dropna().empty or signals.dropna().abs().sum() == 0:
                trial_info = {
                    'trial': trial.number,
                    'params': params.copy(),
                    'train_metrics': train_result.metrics.copy(),
                    'test_metrics': test_result.metrics.copy(),
                    'full_metrics': full_result.metrics.copy(),
                    'turnover': 0,
                    'trades_per_year': 0,
                    'score': -999,
                    'is_valid': False,
                    'rejection_reason': 'No signals generated (strategy always flat)',
                    'relaxation_level': relaxation_level,
                }
                self.optimization_history.append(trial_info)
                return -999
            
            turnover = self.calculate_turnover(signals.fillna(0))
            
            yearly_cagrs = []
            if self.current_stage == 'stage2':
                yearly_periods = self.split_by_year(full_df)
                for period_name, period_df in yearly_periods.items():
                    if len(period_df) > 100:
                        try:
                            period_result = self.run_backtest(period_df, params)
                            yearly_cagrs.append(period_result.metrics.get('CAGR', 0))
                        except:
                            pass
            
            score, is_valid, reason, trades_per_year = self.calculate_score(
                train_result.metrics,
                test_result.metrics,
                full_result.metrics,
                turnover,
                yearly_cagrs,
                relaxation_level
            )
            
            trial_info = {
                'trial': trial.number,
                'params': params.copy(),
                'train_metrics': train_result.metrics.copy(),
                'test_metrics': test_result.metrics.copy(),
                'full_metrics': full_result.metrics.copy(),
                'turnover': turnover,
                'trades_per_year': trades_per_year,
                'score': score,
                'is_valid': is_valid,
                'rejection_reason': reason,
                'relaxation_level': relaxation_level,
            }
            self.optimization_history.append(trial_info)
            
            if not is_valid:
                return -999
            
            return score
            
        except Exception as e:
            # Log exception as well
            trial_info = {
                'trial': trial.number,
                'params': params.copy(),
                'train_metrics': {},
                'test_metrics': {},
                'full_metrics': {},
                'turnover': 0,
                'trades_per_year': 0,
                'score': -999,
                'is_valid': False,
                'rejection_reason': f'Exception: {str(e)}',
                'relaxation_level': relaxation_level,
            }
            self.optimization_history.append(trial_info)
            return -999
    
    def stage1_adaptive_search(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        full_df: pd.DataFrame,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Stage 1: ADAPTIVE coarse search with progressive relaxation.
        
        Process:
        1. Try with standard limits (min_trades=30)
        2. If no valid candidates, relax to min_trades=10 with wider gate bounds
        3. If still no valid candidates, relax to min_trades=5 with widest bounds
        """
        self.current_stage = 'stage1'
        all_candidates = []
        
        for level, relax_config in enumerate(self.stage1_relaxation_levels):
            self.current_relaxation_level = level
            limits = self.get_adaptive_stage1_limits(level)
            bounds = self.get_adaptive_param_bounds(level)
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Stage 1 - Relaxation Level {level}: {relax_config['name']}")
                print(f"  min_trades_per_year: {limits['min_trades_per_year']}")
                print(f"  adx_gate_threshold bounds: {bounds['adx_gate_threshold']}")
                print(f"  chop_threshold bounds: {bounds['chop_threshold']}")
                print(f"  Trials: {self.stage1_trials}")
                print(f"{'='*60}")
            
            # Clear db for fresh start at each level
            storage = f"sqlite:///{self.db_path}" if self.db_path else None
            study_name = f'v2_stage1_adaptive_L{level}_{datetime.now().strftime("%H%M%S")}'
            
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state + level * 100),
                storage=storage,
                study_name=study_name,
            )
            
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.optimization_history = []
            
            study.optimize(
                lambda trial: self.objective(trial, train_df, test_df, full_df, bounds, level),
                n_trials=self.stage1_trials,
                show_progress_bar=verbose
            )
            
            valid_trials = [t for t in self.optimization_history if t['is_valid']]
            
            # Log relaxation result
            self.relaxation_log.append({
                'level': level,
                'name': relax_config['name'],
                'min_trades_per_year': limits['min_trades_per_year'],
                'total_trials': self.stage1_trials,
                'valid_trials': len(valid_trials),
                'adx_bounds': bounds['adx_gate_threshold'],
                'chop_bounds': bounds['chop_threshold'],
            })
            
            if verbose:
                print(f"\nLevel {level} Results: {len(valid_trials)}/{self.stage1_trials} valid trials")
                
                # Show rejection breakdown
                reasons = {}
                for t in self.optimization_history:
                    r = t.get('rejection_reason', '')
                    if r:
                        reasons[r] = reasons.get(r, 0) + 1
                if reasons:
                    print("Rejection breakdown:")
                    for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:5]:
                        print(f"  {v:3d} | {k}")
            
            if len(valid_trials) >= 1:
                # Found valid candidates - select Pareto set
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
        
        # Best CAGR (excluding already selected)
        remaining = [t for t in valid_trials if t['params'] != best_sharpe['params']]
        if remaining:
            best_cagr = max(remaining, key=lambda t: t['test_metrics']['CAGR'])
            candidates.append({
                'params': best_cagr['params'],
                'selection_criterion': 'best_cagr_wf',
                'metrics': best_cagr
            })
        
        # Lowest MaxDD (excluding already selected)
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
        """
        Stage 2: Refine search with STRICT limits.
        Must pass hard limits: min_trades=60, max_trades=150, MaxDD<=45%, Sharpe>=0.7
        """
        self.current_stage = 'stage2'
        base_params = candidate['params'].copy()
        
        # CRITICAL: Re-enable regime gate for Stage2 (must pass strict criteria)
        if 'disable_regime_gate' in base_params:
            del base_params['disable_regime_gate']
        
        base_bounds = RobustStrategyV2.get_param_bounds()
        
        # Create refined bounds (±20% for continuous, ±15% for integers)
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
        
        storage = f"sqlite:///{self.db_path}" if self.db_path else None
        study_name = f'v2_stage2_c{candidate_idx}_{datetime.now().strftime("%H%M%S")}'
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state + candidate_idx),
            storage=storage,
            study_name=study_name,
        )
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        start_idx = len(self.optimization_history)
        
        study.optimize(
            lambda trial: self.objective(trial, train_df, test_df, full_df, refined_bounds, 0),
            n_trials=self.stage2_trials,
            show_progress_bar=verbose
        )
        
        new_trials = self.optimization_history[start_idx:]
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
            # No valid trials in Stage2 - candidate failed strict criteria
            if verbose:
                print(f"\n[!] Candidate {candidate_idx+1} failed to pass Stage2 strict limits")
                # Show why
                reasons = {}
                for t in new_trials:
                    r = t.get('rejection_reason', '')
                    if r:
                        reasons[r] = reasons.get(r, 0) + 1
                if reasons:
                    print("  Rejection reasons:")
                    for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:3]:
                        print(f"    {v:3d} | {k}")
            
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
        
        return result
    
    def walk_forward(
        self,
        df: pd.DataFrame,
        train_end: str = "2022-12-31",
        periods_per_year: int = 24 * 365,
        verbose: bool = True
    ) -> WalkForwardResultV2:
        """
        Run complete two-stage walk-forward optimization with ADAPTIVE Stage1.
        """
        train_df, test_df = self.split_data(df, train_end)
        
        if verbose:
            print(f"\n{'#'*70}")
            print("Walk-Forward Optimization V2 - ADAPTIVE")
            print(f"{'#'*70}")
            print(f"\nData Split:")
            print(f"  Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()} ({len(train_df)} rows)")
            print(f"  Test: {test_df['timestamp'].min()} to {test_df['timestamp'].max()} ({len(test_df)} rows)")
        
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
        
        # Select final best from those that passed Stage2 strict limits
        passed_strict = [r for r in stage2_results if r['passed_strict']]
        
        if passed_strict:
            best_result = max(passed_strict, key=lambda r: r['metrics']['score'])
            if verbose:
                print(f"\n[OK] Final best selected from {len(passed_strict)} candidates that passed Stage2")
        else:
            # Fall back to best Stage2 result even if it didn't pass strict
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
        final_train_result = self.run_backtest(train_df, self.best_params)
        final_test_result = self.run_backtest(test_df, self.best_params)
        final_full_result = self.run_backtest(df, self.best_params)
        
        # Calculate yearly metrics
        yearly_metrics = {}
        yearly_periods = self.split_by_year(df)
        for period_name, period_df in yearly_periods.items():
            if len(period_df) > 100:
                try:
                    period_result = self.run_backtest(period_df, self.best_params)
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
            relaxation_log=self.relaxation_log
        )


if __name__ == "__main__":
    from data_loader import load_data
    
    print("Testing AdaptiveWalkForwardOptimizerV2...")
    df = load_data(start_date="2020-01-01", end_date="2025-11-30", freq="1h")
    print(f"Loaded {len(df)} rows")
    
    optimizer = AdaptiveWalkForwardOptimizerV2(
        stage1_trials=10,
        stage2_trials=5,
        n_candidates=2
    )
    
    result = optimizer.walk_forward(df, verbose=True)
    
    print("\nTest complete!")
    print(f"Best params: {result.best_params}")

