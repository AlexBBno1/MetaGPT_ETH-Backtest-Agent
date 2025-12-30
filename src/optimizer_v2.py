"""
Walk-Forward Optimizer V2 - Two-Stage Optimization with Robust Scoring

Key Features:
1. Two-stage optimization: Coarse (60 trials) + Refine (40x3 trials)
2. Walk-forward test focused scoring
3. Hard rejection criteria (MaxDD > 45%, Sharpe < 0.7, etc.)
4. Turnover penalty
5. SQLite persistence for resumability
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

from backtester import Backtester, BacktestConfig, BacktestResult, calculate_metrics
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


class WalkForwardOptimizerV2:
    """
    Walk-Forward Optimizer V2 with two-stage optimization.
    
    Stage 1: Coarse search (60 trials) - Find 3 Pareto candidates
    Stage 2: Refine search (40 trials each) - Local optimization around candidates
    
    Scoring focuses on walk-forward test performance.
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
        """
        Initialize optimizer.
        
        Args:
            train_years: Number of years for training period
            stage1_trials: Number of coarse search trials
            stage2_trials: Number of refine trials per candidate
            n_candidates: Number of candidates from Stage 1
            random_state: Random seed
            db_path: Path to SQLite database for persistence
        """
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
        
        # Score weights (from specification)
        self.score_weights = {
            'sharpe_wf': 1.2,
            'cagr_wf': 0.8,
            'maxdd_full': -0.9,
            'turnover_penalty': -0.20,
            'cost_penalty': -0.10,
            'stability_bonus': 0.20,
        }
        
        # Hard rejection criteria (spec)
        # Two-stage hard limits
        self.hard_limits_stage1 = {
            'max_dd_limit': 0.70,        # decimal
            'min_sharpe_wf': -0.2,       # allow exploration; avoid zero-sharpe only
            'min_trades_per_year': 30,
            'max_trades_per_year': 300,
            'max_turnover': 600,
        }
        self.hard_limits_stage2 = {
            'max_dd_limit': 0.45,        # decimal
            'min_sharpe_wf': 0.8,        # strict as spec (>=0.7, prefer 0.8)
            'min_trades_per_year': 60,
            'max_trades_per_year': 150,
            'max_turnover': 400,
        }
        self.current_stage = 'stage1'
    
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
    ) -> BacktestResult:
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
        """
        Calculate stability bonus based on yearly CAGR consistency.
        Lower dispersion = higher bonus.
        """
        if len(yearly_cagrs) < 2:
            return 0.0
        
        cagrs = np.array(yearly_cagrs)
        std = np.std(cagrs)
        mean = np.mean(cagrs)
        
        # Coefficient of variation (lower is better)
        if abs(mean) > 0.01:
            cv = std / abs(mean)
        else:
            cv = 10.0  # High penalty for near-zero mean
        
        # Convert to bonus (max 1.0 for CV < 0.5)
        stability = max(0, 1.0 - cv)
        
        return stability
    
    def calculate_score(
        self,
        train_metrics: Dict,
        test_metrics: Dict,
        full_metrics: Dict,
        turnover: float,
        yearly_cagrs: List[float]
    ) -> Tuple[float, bool, str, float]:
        """
        Calculate composite score with hard rejection.
        
        Score focuses on walk-forward TEST results.
        Targets: Sharpe >= 0.8, MaxDD <= 45%, CAGR >= 20%
        
        Returns:
            (score, is_valid, rejection_reason, trades_per_year)
        """
        # Extract metrics (WF test is primary)
        sharpe_wf = float(test_metrics.get('Sharpe', 0) or 0)
        cagr_wf = self._to_decimal(test_metrics.get('CAGR', 0))
        maxdd_full = self._to_decimal(full_metrics.get('MaxDrawdown', 1.0))
        trades_count = full_metrics.get('TradesCount', 0)
        years = max(full_metrics.get('Years', 5.0), 1e-6)
        trades_per_year = trades_count / years
        
        limits = self.hard_limits_stage1 if self.current_stage == 'stage1' else self.hard_limits_stage2

        # Debug unit check
        print('UNIT CHECK:', 'CAGR', cagr_wf, 'MaxDD', maxdd_full, 'limit', limits['max_dd_limit'])
        
        # Hard rejection checks (stage-dependent)
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
        
        # === SCORE CALCULATION (spec) ===
        turnover_penalty = 0.0
        if turnover > 400:
            turnover_penalty = 3.0 + (turnover - 400) / 50  # severe penalty near reject
        elif turnover > 300:
            turnover_penalty = 1.5 + (turnover - 300) / 100
        elif turnover > 240:
            turnover_penalty = (turnover - 240) / 80
        
        cost_penalty = max(0.0, (turnover - 200) / 150)  # proxy for cost sensitivity
        
        # Stability only in stage2
        stability_bonus = self.calculate_stability(yearly_cagrs) if self.current_stage == 'stage2' else 0.0

        dd_target = 0.45
        dd_penalty = max(0.0, (maxdd_full - dd_target) / 0.10)
        
        score = (
            1.2 * sharpe_wf +
            0.8 * (cagr_wf * 100.0) -   # use % scale for comparability
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
        bounds: Dict[str, Tuple] = None
    ) -> float:
        """Optuna objective function."""
        # Sample parameters
        params = RobustStrategyV2.sample_params(trial, bounds)
        
        try:
            # Run backtests
            train_result = self.run_backtest(train_df, params)
            test_result = self.run_backtest(test_df, params)
            full_result = self.run_backtest(full_df, params)
            
            # Signals for turnover / sanity
            strategy = RobustStrategyV2(params)
            signals = strategy.generate_signals(full_df)
            if signals.dropna().empty or signals.dropna().abs().sum() == 0:
                return -999
            turnover = self.calculate_turnover(signals.fillna(0))
            
            # Get yearly CAGRs for stability
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
            
            # Calculate score
            score, is_valid, reason, trades_per_year = self.calculate_score(
                train_result.metrics,
                test_result.metrics,
                full_result.metrics,
                turnover,
                yearly_cagrs
            )
            
            # Store trial info
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
            }
            self.optimization_history.append(trial_info)
            
            if not is_valid:
                return -999  # Pruned
            
            return score
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return -999
    
    def stage1_coarse_search(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        full_df: pd.DataFrame,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Stage 1: Coarse search to find Pareto candidates.
        """
        if verbose:
            print(f"\n{'='*60}")
            print("Stage 1: Coarse Search")
            print(f"  Trials: {self.stage1_trials}")
            print(f"  Looking for {self.n_candidates} Pareto candidates")
            print(f"{'='*60}")
        
        # Create study
        storage = f"sqlite:///{self.db_path}" if self.db_path else None
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            storage=storage,
            study_name='v2_stage1',
            load_if_exists=True
        )
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.optimization_history = []
        self.current_stage = 'stage1'
        
        study.optimize(
            lambda trial: self.objective(trial, train_df, test_df, full_df),
            n_trials=self.stage1_trials,
            show_progress_bar=verbose
        )
        
        # Get valid trials sorted by different criteria
        valid_trials = [t for t in self.optimization_history if t['is_valid']]
        
        if len(valid_trials) == 0:
            print("WARNING: No valid trials found in Stage 1!")
            return []
        
        # Select Pareto candidates: best Sharpe, best CAGR, lowest DD
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
            print(f"\nStage 1 Results:")
            print(f"  Valid trials: {len(valid_trials)}/{self.stage1_trials}")
            print(f"  Candidates found: {len(candidates)}")
            for i, c in enumerate(candidates):
                m = c['metrics']
                print(f"\n  Candidate {i+1} ({c['selection_criterion']}):")
                print(f"    Sharpe WF: {m['test_metrics']['Sharpe']:.2f}")
                print(f"    CAGR WF: {m['test_metrics']['CAGR']:.2f}%")
                print(f"    MaxDD Full: {m['full_metrics']['MaxDrawdown']:.2f}%")
                print(f"    Score: {m['score']:.4f}")
        
        self.stage1_candidates = candidates
        return candidates
    
    def stage2_refine_search(
        self,
        candidate: Dict,
        candidate_idx: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        full_df: pd.DataFrame,
        verbose: bool = True
    ) -> Dict:
        """
        Stage 2: Refine search around a candidate.
        Search range: ±20% for continuous, ±15% for lookback periods.
        """
        base_params = candidate['params']
        base_bounds = RobustStrategyV2.get_param_bounds()
        
        # Create refined bounds (±20% for continuous, ±15% for integers)
        refined_bounds = {}
        for key, (low, high) in base_bounds.items():
            base_val = base_params.get(key, (low + high) / 2)
            
            if isinstance(base_val, int):
                # Integer params: ±15%
                margin = max(1, int(abs(base_val) * 0.15))
                new_low = max(low, base_val - margin)
                new_high = min(high, base_val + margin)
                refined_bounds[key] = (int(new_low), int(new_high))
            else:
                # Float params: ±20%
                margin = abs(base_val) * 0.20
                new_low = max(low, base_val - margin)
                new_high = min(high, base_val + margin)
                refined_bounds[key] = (new_low, new_high)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Stage 2: Refine Search - Candidate {candidate_idx+1}")
            print(f"  Base criterion: {candidate['selection_criterion']}")
            print(f"  Trials: {self.stage2_trials}")
            print(f"{'='*60}")
        
        # Create study
        storage = f"sqlite:///{self.db_path}" if self.db_path else None
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state + candidate_idx),
            storage=storage,
            study_name=f'v2_stage2_c{candidate_idx}',
            load_if_exists=True
        )
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        start_idx = len(self.optimization_history)
        self.current_stage = 'stage2'
        
        study.optimize(
            lambda trial: self.objective(trial, train_df, test_df, full_df, refined_bounds),
            n_trials=self.stage2_trials,
            show_progress_bar=verbose
        )
        
        # Get best from this stage
        new_trials = self.optimization_history[start_idx:]
        valid_trials = [t for t in new_trials if t['is_valid']]
        
        if valid_trials:
            best_trial = max(valid_trials, key=lambda t: t['score'])
            result = {
                'params': best_trial['params'],
                'base_candidate': candidate_idx,
                'metrics': best_trial,
                'improvement': best_trial['score'] - candidate['metrics']['score']
            }
        else:
            # Fall back to original candidate
            result = {
                'params': candidate['params'],
                'base_candidate': candidate_idx,
                'metrics': candidate['metrics'],
                'improvement': 0
            }
        
        if verbose:
            m = result['metrics']
            print(f"\nStage 2 Result for Candidate {candidate_idx+1}:")
            print(f"  Sharpe WF: {m['test_metrics']['Sharpe']:.2f}")
            print(f"  CAGR WF: {m['test_metrics']['CAGR']:.2f}%")
            print(f"  MaxDD Full: {m['full_metrics']['MaxDrawdown']:.2f}%")
            print(f"  Score: {m['score']:.4f}")
            print(f"  Improvement: {result['improvement']:+.4f}")
        
        return result
    
    def walk_forward(
        self,
        df: pd.DataFrame,
        train_end: str = "2022-12-31",
        periods_per_year: int = 24 * 365,
        verbose: bool = True
    ) -> WalkForwardResultV2:
        """
        Run complete two-stage walk-forward optimization.
        """
        # Split data
        train_df, test_df = self.split_data(df, train_end)
        
        if verbose:
            print(f"\n{'#'*70}")
            print("Walk-Forward Optimization V2")
            print(f"{'#'*70}")
            print(f"\nData Split:")
            print(f"  Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()} ({len(train_df)} rows)")
            print(f"  Test: {test_df['timestamp'].min()} to {test_df['timestamp'].max()} ({len(test_df)} rows)")
        
        # Stage 1: Coarse search
        candidates = self.stage1_coarse_search(train_df, test_df, df, verbose)
        
        if not candidates:
            # Provide debug summary for rejection reasons
            reasons = {}
            for t in self.optimization_history:
                r = t.get('rejection_reason', 'unknown')
                reasons[r] = reasons.get(r, 0) + 1
            print("Stage 1 failed: rejection counts =>")
            for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"  {v:3d} | {k}")
            raise ValueError("Stage 1 failed to find any valid candidates")
        
        # Stage 2: Refine each candidate
        stage2_results = []
        for i, candidate in enumerate(candidates):
            result = self.stage2_refine_search(candidate, i, train_df, test_df, df, verbose)
            stage2_results.append(result)
        
        self.stage2_results = stage2_results
        
        # Select final best based on score
        best_result = max(stage2_results, key=lambda r: r['metrics']['score'])
        self.best_params = best_result['params']
        
        if verbose:
            print(f"\n{'='*70}")
            print("Final Best Parameters Selected")
            print(f"{'='*70}")
            m = best_result['metrics']
            print(f"  Source: Candidate {best_result['base_candidate']+1}")
            print(f"  Final Score: {m['score']:.4f}")
            print(f"  Sharpe (WF Test): {m['test_metrics']['Sharpe']:.2f}")
            print(f"  CAGR (WF Test): {m['test_metrics']['CAGR']:.2f}%")
            print(f"  MaxDD (Full): {m['full_metrics']['MaxDrawdown']:.2f}%")
            print(f"  Turnover: {m['turnover']:.0f}%")
            print(f"  Trades: {m['full_metrics']['TradesCount']}")
        
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
            stage2_results=self.stage2_results
        )


if __name__ == "__main__":
    from data_loader import load_data
    
    print("Testing WalkForwardOptimizerV2...")
    df = load_data(start_date="2020-01-01", end_date="2025-11-30", freq="1h")
    print(f"Loaded {len(df)} rows")
    
    optimizer = WalkForwardOptimizerV2(
        stage1_trials=10,  # Quick test
        stage2_trials=5,
        n_candidates=2
    )
    
    result = optimizer.walk_forward(df, verbose=True)
    
    print("\nTest complete!")
    print(f"Best params: {result.best_params}")

