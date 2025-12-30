"""
Walk-Forward Optimization for ETH Empirical Routes.
Train 3Y, Test 2Y methodology.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import optuna
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from backtester import Backtester, BacktestConfig, BacktestResult
from strategies.factor_strategy import FactorStrategy
from strategies.route_configs import get_route_config


@dataclass
class WalkForwardResult:
    """Results from walk-forward optimization."""
    best_params: Dict
    train_metrics: Dict
    test_metrics: Dict
    full_metrics: Dict
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    optimization_history: List[Dict]


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization with train/test split.
    """
    
    def __init__(
        self,
        route: str,
        train_years: int = 3,
        n_trials: int = 100,
        random_state: int = 42
    ):
        """
        Initialize optimizer.
        
        Args:
            route: Route name ('A', 'B', 'C')
            train_years: Number of years for training period
            n_trials: Number of optimization trials
            random_state: Random seed for reproducibility
        """
        self.route = route
        self.route_config = get_route_config(route)
        self.train_years = train_years
        self.n_trials = n_trials
        self.random_state = random_state
        
        self.best_params = None
        self.optimization_history = []
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_end: str = "2022-12-31"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            df: Full dataset
            train_end: End date for training period
            
        Returns:
            (train_df, test_df)
        """
        train_end_dt = pd.to_datetime(train_end)
        
        train_df = df[df['timestamp'] <= train_end_dt].copy().reset_index(drop=True)
        test_df = df[df['timestamp'] > train_end_dt].copy().reset_index(drop=True)
        
        return train_df, test_df
    
    def objective(
        self,
        trial: optuna.Trial,
        df: pd.DataFrame,
        periods_per_year: int = 24 * 365
    ) -> float:
        """
        Optuna objective function for optimization.
        """
        # Get optimized parameters
        params = FactorStrategy.optimize_params(trial, self.route_config)
        
        # Create strategy with updated config
        config = self.route_config.copy()
        
        # Update config with trial params
        config['leverage'] = params['leverage']
        config.setdefault('trend_persistence_params', {})['adx_threshold'] = params['adx_threshold']
        config.setdefault('chop_filter_params', {})['chop_threshold'] = params['chop_threshold']
        config.setdefault('signal_params', {})['momentum_lookback'] = params['momentum_lookback']
        config.setdefault('signal_params', {})['ema_fast'] = params['ema_fast']
        config.setdefault('signal_params', {})['ema_slow'] = params['ema_slow']
        config.setdefault('signal_params', {})['donchian_period'] = params['donchian_period']
        
        # Run backtest
        strategy = FactorStrategy(config)
        signals = strategy.generate_signals(df)
        
        bt_config = BacktestConfig(
            initial_capital=15000.0,
            leverage=params['leverage'],
            max_leverage=config['max_leverage']
        )
        
        backtester = Backtester(bt_config)
        result = backtester.run(df, signals, periods_per_year)
        
        # Get route-specific targets
        risk_params = self.route_config.get('risk_params', {})
        max_dd_target = risk_params.get('max_drawdown_target', 0.50)
        
        # Objective: Maximize Sharpe, penalize if MaxDD exceeds target
        sharpe = result.metrics['Sharpe']
        max_dd = result.metrics['MaxDrawdown'] / 100
        cagr = result.metrics['CAGR'] / 100
        
        # Penalty for exceeding MaxDD target
        dd_penalty = max(0, (max_dd - max_dd_target) * 5)
        
        # Combined objective: Sharpe + CAGR bonus - DD penalty
        objective_value = sharpe + cagr * 0.5 - dd_penalty
        
        # Store trial info
        self.optimization_history.append({
            'trial': trial.number,
            'params': params.copy(),
            'sharpe': sharpe,
            'cagr': result.metrics['CAGR'],
            'max_dd': result.metrics['MaxDrawdown'],
            'objective': objective_value
        })
        
        return objective_value
    
    def optimize(
        self,
        train_df: pd.DataFrame,
        periods_per_year: int = 24 * 365,
        verbose: bool = True
    ) -> Dict:
        """
        Run optimization on training data.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Optimizing Route {self.route}: {self.route_config['name']}")
            print(f"Training data: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
            print(f"Rows: {len(train_df)}")
            print(f"{'='*60}")
        
        # Create Optuna study
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )
        
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Run optimization
        self.optimization_history = []
        study.optimize(
            lambda trial: self.objective(trial, train_df, periods_per_year),
            n_trials=self.n_trials,
            show_progress_bar=verbose
        )
        
        self.best_params = study.best_params
        
        if verbose:
            print(f"\nBest parameters found:")
            for k, v in self.best_params.items():
                print(f"  {k}: {v}")
            print(f"Best objective value: {study.best_value:.4f}")
        
        return self.best_params
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        params: Dict = None,
        periods_per_year: int = 24 * 365
    ) -> BacktestResult:
        """
        Run backtest with specified parameters.
        """
        if params is None:
            params = self.best_params or {}
        
        # Create config with params
        config = self.route_config.copy()
        
        if params:
            config['leverage'] = params.get('leverage', config['leverage'])
            
            if 'adx_threshold' in params:
                config.setdefault('trend_persistence_params', {})['adx_threshold'] = params['adx_threshold']
            if 'chop_threshold' in params:
                config.setdefault('chop_filter_params', {})['chop_threshold'] = params['chop_threshold']
            if 'momentum_lookback' in params:
                config.setdefault('signal_params', {})['momentum_lookback'] = params['momentum_lookback']
            if 'ema_fast' in params:
                config.setdefault('signal_params', {})['ema_fast'] = params['ema_fast']
            if 'ema_slow' in params:
                config.setdefault('signal_params', {})['ema_slow'] = params['ema_slow']
            if 'donchian_period' in params:
                config.setdefault('signal_params', {})['donchian_period'] = params['donchian_period']
        
        # Create and run strategy
        strategy = FactorStrategy(config)
        signals = strategy.generate_signals(df)
        
        bt_config = BacktestConfig(
            initial_capital=15000.0,
            leverage=config['leverage'],
            max_leverage=config['max_leverage']
        )
        
        backtester = Backtester(bt_config)
        return backtester.run(df, signals, periods_per_year)
    
    def walk_forward(
        self,
        df: pd.DataFrame,
        train_end: str = "2022-12-31",
        periods_per_year: int = 24 * 365,
        verbose: bool = True
    ) -> WalkForwardResult:
        """
        Run complete walk-forward optimization.
        
        1. Split data into train/test
        2. Optimize on train
        3. Validate on test
        4. Run on full period
        """
        # Split data
        train_df, test_df = self.split_data(df, train_end)
        
        if verbose:
            print(f"\nData Split:")
            print(f"  Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()} ({len(train_df)} rows)")
            print(f"  Test: {test_df['timestamp'].min()} to {test_df['timestamp'].max()} ({len(test_df)} rows)")
        
        # Optimize on training data
        best_params = self.optimize(train_df, periods_per_year, verbose)
        
        # Backtest on training data (IS)
        train_result = self.run_backtest(train_df, best_params, periods_per_year)
        
        if verbose:
            print(f"\nTraining (In-Sample) Results:")
            print(f"  CAGR: {train_result.metrics['CAGR']:.2f}%")
            print(f"  MaxDD: {train_result.metrics['MaxDrawdown']:.2f}%")
            print(f"  Sharpe: {train_result.metrics['Sharpe']:.2f}")
        
        # Backtest on test data (OOS)
        test_result = self.run_backtest(test_df, best_params, periods_per_year)
        
        if verbose:
            print(f"\nTest (Out-of-Sample) Results:")
            print(f"  CAGR: {test_result.metrics['CAGR']:.2f}%")
            print(f"  MaxDD: {test_result.metrics['MaxDrawdown']:.2f}%")
            print(f"  Sharpe: {test_result.metrics['Sharpe']:.2f}")
        
        # Run on full period
        full_result = self.run_backtest(df, best_params, periods_per_year)
        
        if verbose:
            print(f"\nFull Period (2020-2025) Results:")
            print(f"  CAGR: {full_result.metrics['CAGR']:.2f}%")
            print(f"  MaxDD: {full_result.metrics['MaxDrawdown']:.2f}%")
            print(f"  Sharpe: {full_result.metrics['Sharpe']:.2f}")
            print(f"  Final Equity: ${full_result.metrics['FinalEquity']:,.2f}")
        
        return WalkForwardResult(
            best_params=best_params,
            train_metrics=train_result.metrics,
            test_metrics=test_result.metrics,
            full_metrics=full_result.metrics,
            equity_curve=full_result.equity_curve,
            trades=full_result.trades,
            optimization_history=self.optimization_history
        )


def run_route_optimization(
    route: str,
    df: pd.DataFrame,
    train_end: str = "2022-12-31",
    n_trials: int = 100,
    verbose: bool = True
) -> WalkForwardResult:
    """
    Convenience function to run optimization for a route.
    """
    optimizer = WalkForwardOptimizer(
        route=route,
        n_trials=n_trials
    )
    
    return optimizer.walk_forward(df, train_end, verbose=verbose)


if __name__ == "__main__":
    from data_loader import load_data
    
    # Test optimization
    print("Loading data...")
    df = load_data(start_date="2020-01-01", end_date="2025-11-30", freq="1h")
    print(f"Loaded {len(df)} rows")
    
    # Run quick test with fewer trials
    result = run_route_optimization('A', df, n_trials=10, verbose=True)
    
    print("\nOptimization complete!")
    print(f"Best params: {result.best_params}")

