"""
ETH Empirical Routes 5Y Backtest - Main Execution Script

This script runs the complete 5-year empirical backtest for ETH strategies
across three routes (A: Conservative, B: Balanced, C: Aggressive).

Usage:
    python run_backtest.py
    python run_backtest.py --route A
    python run_backtest.py --n-trials 50
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data_loader import load_data, get_data_info
from optimizer import WalkForwardOptimizer, run_route_optimization
from visualizer import (
    plot_equity_curve, 
    plot_routes_comparison,
    plot_yearly_returns,
    generate_report
)


def setup_logging(log_dir: Path):
    """Setup logging to file."""
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    return log_file


def save_results(results: dict, output_dir: Path):
    """Save all results to files."""
    output_dir.mkdir(exist_ok=True)
    
    for route, result in results.items():
        # Save metrics as JSON
        metrics_path = output_dir / f"metrics_route_{route}.json"
        with open(metrics_path, 'w') as f:
            json.dump(result['metrics'], f, indent=2)
        print(f"Saved: {metrics_path}")
        
        # Save equity curve as parquet
        equity_path = output_dir / f"equity_curve_route_{route}.parquet"
        result['equity_curve'].to_parquet(equity_path, index=False)
        print(f"Saved: {equity_path}")
        
        # Save trades as CSV
        trades_path = output_dir / f"trades_route_{route}.csv"
        result['trades'].to_csv(trades_path, index=False)
        print(f"Saved: {trades_path}")
        
        # Save best params
        params_path = output_dir / f"best_params_route_{route}.json"
        with open(params_path, 'w') as f:
            json.dump(result['best_params'], f, indent=2)
        print(f"Saved: {params_path}")
        
        # Generate individual plot
        plot_path = output_dir / f"plot_route_{route}.png"
        plot_equity_curve(result['equity_curve'], route, result['metrics'], plot_path)
    
    # Generate comparison plots
    comparison_path = output_dir / "routes_comparison.png"
    plot_routes_comparison(results, comparison_path)
    
    yearly_path = output_dir / "yearly_returns.png"
    plot_yearly_returns(results, yearly_path)
    
    # Generate report
    report_path = output_dir / "empirical_routes_report.md"
    generate_report(results, report_path)


def run_single_route(route: str, df, n_trials: int = 100, verbose: bool = True) -> dict:
    """Run backtest for a single route."""
    print(f"\n{'='*70}")
    print(f"Running Route {route}")
    print(f"{'='*70}")
    
    result = run_route_optimization(
        route=route,
        df=df,
        train_end="2022-12-31",
        n_trials=n_trials,
        verbose=verbose
    )
    
    return {
        'metrics': result.full_metrics,
        'equity_curve': result.equity_curve,
        'trades': result.trades,
        'best_params': result.best_params,
        'train_metrics': result.train_metrics,
        'test_metrics': result.test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='ETH Empirical Routes 5Y Backtest')
    parser.add_argument('--route', type=str, default=None, 
                        help='Run specific route (A, B, C) or all if not specified')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of optimization trials per route')
    parser.add_argument('--no-log', action='store_true',
                        help='Disable logging to file')
    args = parser.parse_args()
    
    # Setup paths
    output_dir = PROJECT_ROOT / 'outputs'
    log_dir = PROJECT_ROOT / 'logs'
    
    # Setup logging
    if not args.no_log:
        log_file = setup_logging(log_dir)
        print(f"Logging to: {log_file}")
    
    print("="*70)
    print("ETH Empirical Routes 5Y Backtest")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {output_dir}")
    print(f"Optimization Trials: {args.n_trials}")
    print("="*70)
    
    # Load data
    print("\n[1/4] Loading ETH data (2020-01-01 to 2025-11-30)...")
    try:
        df = load_data(
            start_date="2020-01-01",
            end_date="2025-11-30",
            freq="1h"
        )
        
        info = get_data_info(df)
        print(f"Data loaded successfully:")
        print(f"  Rows: {info['rows']}")
        print(f"  Period: {info['start_date']} to {info['end_date']}")
        print(f"  Price range: ${info['price_range'][0]:.2f} - ${info['price_range'][1]:.2f}")
        print(f"  Missing: {info['missing_pct']:.4f}%")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Attempting to fetch from yfinance...")
        df = load_data(
            start_date="2020-01-01",
            end_date="2025-11-30",
            freq="1h",
            force_refresh=True
        )
    
    # Determine routes to run
    if args.route:
        routes = [args.route.upper()]
    else:
        routes = ['A', 'B', 'C']
    
    # Run backtests
    print(f"\n[2/4] Running walk-forward optimization for routes: {routes}")
    results = {}
    
    for route in routes:
        try:
            results[route] = run_single_route(
                route=route,
                df=df,
                n_trials=args.n_trials,
                verbose=True
            )
        except Exception as e:
            print(f"Error running route {route}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("No results generated. Exiting.")
        return
    
    # Save results
    print(f"\n[3/4] Saving results...")
    save_results(results, output_dir)
    
    # Print summary
    print(f"\n[4/4] Summary")
    print("="*70)
    print(f"{'Route':<10} {'CAGR':<10} {'MaxDD':<10} {'Sharpe':<10} {'Final Equity':<15}")
    print("-"*70)
    
    for route in ['A', 'B', 'C']:
        if route in results:
            m = results[route]['metrics']
            print(f"Route {route:<5} {m['CAGR']:<10.2f}% {m['MaxDrawdown']:<10.2f}% "
                  f"{m['Sharpe']:<10.2f} ${m['FinalEquity']:>13,.0f}")
    
    print("="*70)
    
    # Determine best route
    if len(results) > 1:
        best_sharpe = max(results.keys(), key=lambda r: results[r]['metrics']['Sharpe'])
        best_cagr = max(results.keys(), key=lambda r: results[r]['metrics']['CAGR'])
        best_calmar = max(results.keys(), key=lambda r: results[r]['metrics']['Calmar'])
        
        print(f"\nBest Risk-Adjusted (Sharpe): Route {best_sharpe}")
        print(f"Best Returns (CAGR): Route {best_cagr}")
        print(f"Best Risk-Return (Calmar): Route {best_calmar}")
    
    print(f"\nBacktest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

