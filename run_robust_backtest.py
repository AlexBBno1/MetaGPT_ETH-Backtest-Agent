"""
ETH Empirical Routes 5Y - Robust Backtest (No Optimization)

This script runs a more robust backtest using fixed parameters
to avoid overfitting issues seen in the optimized version.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data_loader import load_data, get_data_info
from backtester import Backtester, BacktestConfig
from strategies.robust_strategy import RobustTrendStrategy
from visualizer import (
    plot_equity_curve, 
    plot_routes_comparison,
    plot_yearly_returns,
    generate_report
)


def run_route(route: str, df, verbose: bool = True) -> dict:
    """Run backtest for a single route with fixed params."""
    config = RobustTrendStrategy.get_route_config(route)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {config['name']}")
        print(f"Leverage: {config['leverage']}, Base Size: {config['base_position_size']}")
        print(f"{'='*60}")
    
    strategy = RobustTrendStrategy(config)
    signals = strategy.generate_signals(df)
    
    # Split for validation
    train_end = "2022-12-31"
    train_df = df[df['timestamp'] <= train_end].copy().reset_index(drop=True)
    test_df = df[df['timestamp'] > train_end].copy().reset_index(drop=True)
    
    bt_config = BacktestConfig(
        initial_capital=15000.0,
        leverage=config['leverage'],
        max_leverage=config['max_leverage']
    )
    
    backtester = Backtester(bt_config)
    
    # Train period
    train_signals = signals.iloc[:len(train_df)]
    train_result = backtester.run(train_df, train_signals)
    
    # Test period
    test_signals = signals.iloc[len(train_df):].reset_index(drop=True)
    test_bt = Backtester(BacktestConfig(
        initial_capital=train_result.equity_curve['equity'].iloc[-1],  # Continue from train end
        leverage=config['leverage'],
        max_leverage=config['max_leverage']
    ))
    test_result = test_bt.run(test_df, test_signals)
    
    # Full period
    full_result = backtester.run(df, signals)
    
    if verbose:
        print(f"\nTraining (2020-2022) Results:")
        print(f"  CAGR: {train_result.metrics['CAGR']:.2f}%")
        print(f"  MaxDD: {train_result.metrics['MaxDrawdown']:.2f}%")
        print(f"  Sharpe: {train_result.metrics['Sharpe']:.2f}")
        
        print(f"\nTest (2023-2025) Results:")
        print(f"  CAGR: {test_result.metrics['CAGR']:.2f}%")
        print(f"  MaxDD: {test_result.metrics['MaxDrawdown']:.2f}%")
        print(f"  Sharpe: {test_result.metrics['Sharpe']:.2f}")
        
        print(f"\nFull Period (2020-2025) Results:")
        print(f"  CAGR: {full_result.metrics['CAGR']:.2f}%")
        print(f"  MaxDD: {full_result.metrics['MaxDrawdown']:.2f}%")
        print(f"  Sharpe: {full_result.metrics['Sharpe']:.2f}")
        print(f"  Final Equity: ${full_result.metrics['FinalEquity']:,.2f}")
    
    return {
        'metrics': full_result.metrics,
        'equity_curve': full_result.equity_curve,
        'trades': full_result.trades,
        'train_metrics': train_result.metrics,
        'test_metrics': test_result.metrics,
        'config': config,
        'best_params': config['params'],
    }


def save_results(results: dict, output_dir: Path):
    """Save all results."""
    output_dir.mkdir(exist_ok=True)
    
    for route, result in results.items():
        # Metrics JSON
        metrics_path = output_dir / f"metrics_route_{route}.json"
        with open(metrics_path, 'w') as f:
            json.dump(result['metrics'], f, indent=2)
        print(f"Saved: {metrics_path}")
        
        # Equity parquet
        equity_path = output_dir / f"equity_curve_route_{route}.parquet"
        result['equity_curve'].to_parquet(equity_path, index=False)
        print(f"Saved: {equity_path}")
        
        # Trades CSV
        trades_path = output_dir / f"trades_route_{route}.csv"
        result['trades'].to_csv(trades_path, index=False)
        print(f"Saved: {trades_path}")
        
        # Params JSON
        params_path = output_dir / f"best_params_route_{route}.json"
        with open(params_path, 'w') as f:
            json.dump(result['best_params'], f, indent=2)
        print(f"Saved: {params_path}")
        
        # Plot
        plot_path = output_dir / f"plot_route_{route}.png"
        plot_equity_curve(result['equity_curve'], route, result['metrics'], plot_path)
    
    # Comparison plots
    plot_routes_comparison(results, output_dir / "routes_comparison.png")
    plot_yearly_returns(results, output_dir / "yearly_returns.png")
    
    # Report
    generate_report(results, output_dir / "empirical_routes_report.md")


def main():
    print("="*70)
    print("ETH Empirical Routes 5Y - Robust Backtest")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    output_dir = PROJECT_ROOT / 'outputs'
    
    # Load data
    print("\n[1/3] Loading ETH data...")
    df = load_data(start_date="2020-01-01", end_date="2025-11-30", freq="1h")
    info = get_data_info(df)
    print(f"  Rows: {info['rows']}")
    print(f"  Period: {info['start_date']} to {info['end_date']}")
    
    # Run routes
    print("\n[2/3] Running backtests...")
    results = {}
    for route in ['A', 'B', 'C']:
        results[route] = run_route(route, df, verbose=True)
    
    # Save
    print("\n[3/3] Saving results...")
    save_results(results, output_dir)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Route':<12} {'CAGR':<10} {'MaxDD':<10} {'Sharpe':<10} {'Final':<15}")
    print("-"*70)
    
    for route in ['A', 'B', 'C']:
        m = results[route]['metrics']
        print(f"Route {route:<6} {m['CAGR']:>7.2f}% {m['MaxDrawdown']:>8.2f}% "
              f"{m['Sharpe']:>8.2f} ${m['FinalEquity']:>12,.0f}")
    
    print("="*70)
    
    # Best route analysis
    best_sharpe = max(results.keys(), key=lambda r: results[r]['metrics']['Sharpe'])
    best_cagr = max(results.keys(), key=lambda r: results[r]['metrics']['CAGR'])
    
    print(f"\n✓ Best Risk-Adjusted (Sharpe): Route {best_sharpe}")
    print(f"✓ Best Returns (CAGR): Route {best_cagr}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

