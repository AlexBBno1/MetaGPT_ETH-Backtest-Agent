"""
V2.5 Detailed Export Report
Full trade analysis and comparison with V2
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
from openpyxl import load_workbook
from openpyxl.chart import LineChart, BarChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows
import warnings
warnings.filterwarnings('ignore')

import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src.backtester import Backtester, BacktestConfig

# Import V2.5 strategy
from run_v2_5_stability import StabilityStrategyV25, calculate_rolling_metrics


def main():
    print("="*70)
    print("V2.5 DETAILED EXPORT REPORT")
    print("="*70)
    
    output_dir = PROJECT_ROOT / 'outputs_v2_5'
    output_dir.mkdir(exist_ok=True)
    
    # Load both parameter sets
    v2_params_path = PROJECT_ROOT / 'outputs_v2' / 'best_params_v2.json'
    v25_params_path = output_dir / 'best_params_v25.json'
    
    with open(v2_params_path) as f:
        v2_params = json.load(f)
    with open(v25_params_path) as f:
        v25_params = json.load(f)
    
    print(f"\nV2 Parameters:")
    for k, v in v2_params.items():
        print(f"  {k}: {v}")
    
    print(f"\nV2.5 Parameters:")
    for k, v in v25_params.items():
        if v != v2_params.get(k):
            print(f"  {k}: {v} (was {v2_params.get(k)})")
        else:
            print(f"  {k}: {v}")
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_data(start_date="2020-01-01", end_date="2025-11-30", freq="1h")
    print(f"  Rows: {len(df):,}")
    
    # Generate signals for both versions
    print("\n[2/5] Generating signals...")
    
    # V2 signals
    strategy_v2 = StabilityStrategyV25({k: v2_params.get(k, v) for k, v in StabilityStrategyV25.V2_BEST.items()})
    strategy_v2.params = v2_params  # Override with full V2 params
    signals_v2 = strategy_v2.generate_signals(df)
    
    # V2.5 signals
    strategy_v25 = StabilityStrategyV25({k: v25_params.get(k, v) for k, v in StabilityStrategyV25.V2_BEST.items()})
    strategy_v25.params = v25_params
    signals_v25 = strategy_v25.generate_signals(df)
    
    print(f"  V2 signal range: [{signals_v2.min():.2f}, {signals_v2.max():.2f}]")
    print(f"  V2.5 signal range: [{signals_v25.min():.2f}, {signals_v25.max():.2f}]")
    
    # Run backtests
    print("\n[3/5] Running backtests...")
    
    bt_config = BacktestConfig(
        initial_capital=15000.0,
        commission_bps=4.0,
        slippage_bps=5.0,
        leverage=v25_params.get('leverage_cap', 1.5),
        max_leverage=v25_params.get('leverage_cap', 1.5)
    )
    
    backtester = Backtester(bt_config)
    result_v2 = backtester.run(df.copy(), signals_v2)
    result_v25 = backtester.run(df.copy(), signals_v25)
    
    print(f"  V2:   CAGR={result_v2.metrics['CAGR']:.1f}%, Sharpe={result_v2.metrics['Sharpe']:.2f}, MaxDD={result_v2.metrics['MaxDrawdown']:.1f}%")
    print(f"  V2.5: CAGR={result_v25.metrics['CAGR']:.1f}%, Sharpe={result_v25.metrics['Sharpe']:.2f}, MaxDD={result_v25.metrics['MaxDrawdown']:.1f}%")
    
    # Calculate rolling metrics
    print("\n[4/5] Calculating rolling metrics...")
    rolling_v2 = calculate_rolling_metrics(df, signals_v2, v2_params, window_months=12, step_months=1)
    rolling_v25 = calculate_rolling_metrics(df, signals_v25, v25_params, window_months=12, step_months=1)
    
    # Process trades
    print("\n[5/5] Processing trades and generating reports...")
    
    trades_v25 = result_v25.trades.copy()
    if len(trades_v25) > 0:
        trades_v25['trade_no'] = range(1, len(trades_v25) + 1)
        trades_v25['leverage_used'] = trades_v25['size'] * trades_v25['entry_price'] / bt_config.initial_capital
        trades_v25['notional_value'] = trades_v25['entry_price'] * trades_v25['size']
        trades_v25['pnl_pct'] = trades_v25['return'] * 100
        trades_v25['cumulative_pnl'] = trades_v25['pnl'].cumsum()
        trades_v25['account_balance'] = bt_config.initial_capital + trades_v25['cumulative_pnl']
        
        # Capital invested percentage
        trades_v25['capital_invested_pct'] = (
            trades_v25['notional_value'] / 
            trades_v25['account_balance'].shift(1).fillna(bt_config.initial_capital)
        ) * 100
        
        # Win/loss streaks
        trades_v25['pnl_positive'] = trades_v25['pnl'] > 0
        trades_v25['streak_id'] = (trades_v25['pnl_positive'] != trades_v25['pnl_positive'].shift()).cumsum()
        trades_v25['streak_count'] = trades_v25.groupby('streak_id').cumcount() + 1
        trades_v25['streak_type'] = trades_v25['pnl_positive'].apply(lambda x: 'W' if x else 'L')
        trades_v25['streak'] = trades_v25['streak_type'] + trades_v25['streak_count'].astype(str)
        
        max_win_streak = trades_v25[trades_v25['pnl_positive']]['streak_count'].max() if not trades_v25[trades_v25['pnl_positive']].empty else 0
        max_loss_streak = trades_v25[~trades_v25['pnl_positive']]['streak_count'].max() if not trades_v25[~trades_v25['pnl_positive']].empty else 0
    else:
        max_win_streak = 0
        max_loss_streak = 0
    
    # Generate comparison data
    comparison_data = []
    metrics_to_compare = [
        ('CAGR', '%'), ('MaxDrawdown', '%'), ('Sharpe', ''), 
        ('TradesCount', ''), ('WinRate', '%'), ('AvgWin', '%'),
        ('AvgLoss', '%'), ('FinalEquity', '$')
    ]
    
    for metric, unit in metrics_to_compare:
        v2_val = result_v2.metrics.get(metric, 0)
        v25_val = result_v25.metrics.get(metric, 0)
        diff = v25_val - v2_val
        
        comparison_data.append({
            'Metric': metric,
            'V2': f"{v2_val:.2f}{unit}" if unit != '$' else f"${v2_val:,.2f}",
            'V2.5': f"{v25_val:.2f}{unit}" if unit != '$' else f"${v25_val:,.2f}",
            'Diff': f"{diff:+.2f}{unit}" if unit != '$' else f"${diff:+,.2f}",
            'Better': 'V2.5' if (diff > 0 and metric != 'MaxDrawdown' and metric != 'AvgLoss') or 
                              (diff < 0 and (metric == 'MaxDrawdown' or metric == 'AvgLoss')) else 'V2'
        })
    
    # Generate Excel report
    excel_path = output_dir / 'v25_full_report.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 1. Summary comparison
        pd.DataFrame(comparison_data).to_excel(writer, sheet_name='V2 vs V2.5 Summary', index=False)
        
        # 2. V2.5 Parameters
        params_df = pd.DataFrame([
            {'Parameter': k, 'V2': v2_params.get(k, 'N/A'), 'V2.5': v25_params.get(k, 'N/A'),
             'Changed': 'YES' if v2_params.get(k) != v25_params.get(k) else ''}
            for k in set(list(v2_params.keys()) + list(v25_params.keys()))
        ])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)
        
        # 3. Rolling Performance
        rolling_comparison = pd.DataFrame({
            'Window': range(1, min(len(rolling_v2), len(rolling_v25)) + 1),
            'V2_CAGR': rolling_v2['CAGR'].values[:min(len(rolling_v2), len(rolling_v25))],
            'V25_CAGR': rolling_v25['CAGR'].values[:min(len(rolling_v2), len(rolling_v25))],
            'V2_MaxDD': rolling_v2['MaxDD'].values[:min(len(rolling_v2), len(rolling_v25))],
            'V25_MaxDD': rolling_v25['MaxDD'].values[:min(len(rolling_v2), len(rolling_v25))],
        })
        rolling_comparison.to_excel(writer, sheet_name='Rolling 12M', index=False)
        
        # 4. All Trades (V2.5)
        if len(trades_v25) > 0:
            trade_cols = [
                'trade_no', 'entry_time', 'exit_time', 'side',
                'entry_price', 'exit_price', 'size', 'leverage_used',
                'notional_value', 'capital_invested_pct', 'pnl', 'pnl_pct', 
                'cumulative_pnl', 'account_balance', 'streak',
                'holding_days', 'fees'
            ]
            existing_cols = [c for c in trade_cols if c in trades_v25.columns]
            trades_export = trades_v25[existing_cols].copy()
            trades_export.to_excel(writer, sheet_name='V2.5 All Trades', index=False)
        
        # 5. Monthly Performance
        equity_v25 = result_v25.equity_curve.copy()
        if isinstance(equity_v25, pd.DataFrame):
            equity_v25 = equity_v25.iloc[:, 0]  # Get first column if DataFrame
        equity_v25.index = pd.to_datetime(equity_v25.index)
        
        # Convert to numeric series properly
        equity_series = pd.Series(equity_v25.values.astype(float), index=equity_v25.index)
        monthly = equity_series.resample('M').last()
        
        # Calculate returns manually to avoid type issues
        monthly_values = monthly.values
        monthly_returns_values = (monthly_values[1:] / monthly_values[:-1] - 1) * 100
        
        monthly_df = pd.DataFrame({
            'Month': monthly.index[1:].strftime('%Y-%m'),
            'Return_Pct': monthly_returns_values,
            'Equity': monthly_values[1:]
        })
        monthly_df.to_excel(writer, sheet_name='Monthly Returns', index=False)
        
        # 6. Yearly Summary
        yearly = equity_series.resample('Y').last()
        yearly_values = yearly.values
        yearly_returns_values = (yearly_values[1:] / yearly_values[:-1] - 1) * 100
        
        yearly_df = pd.DataFrame({
            'Year': yearly.index[1:].year,
            'Return_Pct': yearly_returns_values,
            'Equity': yearly_values[1:]
        })
        yearly_df.to_excel(writer, sheet_name='Yearly Summary', index=False)
        
        # 7. Streak Summary
        streak_summary = pd.DataFrame([
            {'Metric': 'Max Win Streak', 'Value': max_win_streak},
            {'Metric': 'Max Loss Streak', 'Value': max_loss_streak},
            {'Metric': 'Total Trades', 'Value': len(trades_v25)},
            {'Metric': 'Win Rate', 'Value': f"{result_v25.metrics['WinRate']:.1f}%"},
        ])
        streak_summary.to_excel(writer, sheet_name='Streak Summary', index=False)
    
    print(f"  Saved: {excel_path.name}")
    
    # Generate comparison charts
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Equity Curve Comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    
    equity_v2 = result_v2.equity_curve
    equity_v25 = result_v25.equity_curve
    
    if isinstance(equity_v2, pd.DataFrame):
        equity_v2 = equity_v2.iloc[:, 0]
    if isinstance(equity_v25, pd.DataFrame):
        equity_v25 = equity_v25.iloc[:, 0]
    
    ax.plot(equity_v2.index, equity_v2.values, 'b-', linewidth=1.5, label=f'V2 (CAGR: {result_v2.metrics["CAGR"]:.1f}%)', alpha=0.8)
    ax.plot(equity_v25.index, equity_v25.values, 'g-', linewidth=1.5, label=f'V2.5 (CAGR: {result_v25.metrics["CAGR"]:.1f}%)', alpha=0.8)
    ax.axhline(y=bt_config.initial_capital, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_ylabel('Equity ($)')
    ax.set_title('V2 vs V2.5 Equity Curve Comparison', fontweight='bold', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Format y-axis
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'equity_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: equity_comparison.png")
    
    # 2. Drawdown Comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Calculate drawdown from equity curve
    def calculate_drawdown(equity):
        if isinstance(equity, pd.DataFrame):
            equity = equity.iloc[:, 0]
        eq_values = equity.values.astype(float)
        running_max = np.maximum.accumulate(eq_values)
        drawdown = (eq_values - running_max) / running_max * 100
        return pd.Series(drawdown, index=equity.index)
    
    dd_v2 = calculate_drawdown(result_v2.equity_curve)
    dd_v25 = calculate_drawdown(result_v25.equity_curve)
    
    axes[0].fill_between(dd_v2.index, 0, dd_v2.values, color='blue', alpha=0.5, label='V2')
    axes[0].set_ylabel('Drawdown (%)')
    axes[0].set_title('V2 Drawdown', fontweight='bold')
    axes[0].axhline(y=-45, color='red', linestyle='--', alpha=0.5, label='45% limit')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-60, 5)
    
    axes[1].fill_between(dd_v25.index, 0, dd_v25.values, color='green', alpha=0.5, label='V2.5')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_title('V2.5 Drawdown', fontweight='bold')
    axes[1].axhline(y=-45, color='red', linestyle='--', alpha=0.5, label='45% limit')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-60, 5)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'drawdown_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: drawdown_comparison.png")
    
    # 3. Rolling CAGR Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(rolling_v2['CAGR'], bins=20, color='blue', alpha=0.7, label='V2', edgecolor='white')
    axes[0].hist(rolling_v25['CAGR'], bins=20, color='green', alpha=0.5, label='V2.5', edgecolor='white')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0].axvline(x=10, color='orange', linestyle='--', alpha=0.7, label='10% target')
    axes[0].set_xlabel('Rolling 12M CAGR (%)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Rolling CAGR Distribution', fontweight='bold')
    axes[0].legend()
    
    axes[1].hist(rolling_v2['MaxDD'], bins=20, color='blue', alpha=0.7, label='V2', edgecolor='white')
    axes[1].hist(rolling_v25['MaxDD'], bins=20, color='green', alpha=0.5, label='V2.5', edgecolor='white')
    axes[1].axvline(x=45, color='red', linestyle='--', alpha=0.7, label='45% limit')
    axes[1].set_xlabel('Rolling 12M MaxDD (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Rolling MaxDD Distribution', fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'rolling_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: rolling_distribution.png")
    
    # Print final summary
    print("\n" + "="*70)
    print("V2 vs V2.5 FINAL COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'V2':>15} {'V2.5':>15} {'Change':>15}")
    print("-"*70)
    print(f"{'CAGR':<25} {result_v2.metrics['CAGR']:>14.1f}% {result_v25.metrics['CAGR']:>14.1f}% {result_v25.metrics['CAGR']-result_v2.metrics['CAGR']:>+14.1f}%")
    print(f"{'MaxDrawdown':<25} {result_v2.metrics['MaxDrawdown']:>14.1f}% {result_v25.metrics['MaxDrawdown']:>14.1f}% {result_v25.metrics['MaxDrawdown']-result_v2.metrics['MaxDrawdown']:>+14.1f}%")
    print(f"{'Sharpe Ratio':<25} {result_v2.metrics['Sharpe']:>15.2f} {result_v25.metrics['Sharpe']:>15.2f} {result_v25.metrics['Sharpe']-result_v2.metrics['Sharpe']:>+15.2f}")
    print(f"{'Final Equity':<25} ${result_v2.metrics['FinalEquity']:>13,.0f} ${result_v25.metrics['FinalEquity']:>13,.0f} ${result_v25.metrics['FinalEquity']-result_v2.metrics['FinalEquity']:>+13,.0f}")
    print(f"{'Trades':<25} {result_v2.metrics['TradesCount']:>15.0f} {result_v25.metrics['TradesCount']:>15.0f}")
    print(f"{'Win Rate':<25} {result_v2.metrics['WinRate']:>14.1f}% {result_v25.metrics['WinRate']:>14.1f}%")
    
    print(f"\n### Rolling Stability ###")
    print(f"{'Median 12M CAGR':<25} {rolling_v2['CAGR'].median():>14.1f}% {rolling_v25['CAGR'].median():>14.1f}%")
    print(f"{'Min 12M CAGR':<25} {rolling_v2['CAGR'].min():>14.1f}% {rolling_v25['CAGR'].min():>14.1f}%")
    print(f"{'Max 12M CAGR':<25} {rolling_v2['CAGR'].max():>14.1f}% {rolling_v25['CAGR'].max():>14.1f}%")
    print(f"{'CAGR Std Dev':<25} {rolling_v2['CAGR'].std():>14.1f}% {rolling_v25['CAGR'].std():>14.1f}%")
    
    print(f"\n### Streak Analysis (V2.5) ###")
    print(f"  Max Win Streak: {max_win_streak}")
    print(f"  Max Loss Streak: {max_loss_streak}")
    
    print(f"\n### WHY V2.5 IS MORE STABLE ###")
    
    reasons = []
    if result_v25.metrics['MaxDrawdown'] < result_v2.metrics['MaxDrawdown']:
        reasons.append(f"1. Lower MaxDrawdown: {result_v25.metrics['MaxDrawdown']:.1f}% vs {result_v2.metrics['MaxDrawdown']:.1f}%")
    
    if result_v25.metrics['Sharpe'] > result_v2.metrics['Sharpe']:
        reasons.append(f"2. Better Sharpe Ratio: {result_v25.metrics['Sharpe']:.2f} vs {result_v2.metrics['Sharpe']:.2f}")
    
    if rolling_v25['CAGR'].std() < rolling_v2['CAGR'].std():
        reasons.append(f"3. More consistent returns (lower std): {rolling_v25['CAGR'].std():.1f}% vs {rolling_v2['CAGR'].std():.1f}%")
    
    if rolling_v25['MaxDD'].max() < rolling_v2['MaxDD'].max():
        reasons.append(f"4. Lower worst-case rolling MaxDD: {rolling_v25['MaxDD'].max():.1f}% vs {rolling_v2['MaxDD'].max():.1f}%")
    
    for reason in reasons:
        print(f"  {reason}")
    
    if not reasons:
        print("  V2.5 focuses on tail risk reduction with similar performance profile")
    
    print(f"\n### OUTPUT FILES ###")
    print(f"  - {excel_path}")
    for f in plots_dir.glob('*.png'):
        print(f"  - {f}")
    
    print("\n" + "="*70)
    print("V2.5 REPORT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
