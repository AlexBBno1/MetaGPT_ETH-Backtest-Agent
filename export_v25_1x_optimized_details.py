"""
V2.5_1x Optimized - Full Trade Details Export
Same format as V2 trade_report
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


class Strategy1xOptimized:
    """Strategy optimized for 1x leverage."""
    
    def __init__(self, params: dict):
        self.params = params
        self.params['leverage_cap'] = 1.0
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params
        
        entry_period = p['donchian_entry_period']
        exit_period = p['donchian_exit_period']
        
        df['don_entry_upper'] = df['high'].rolling(entry_period).max()
        df['don_entry_lower'] = df['low'].rolling(entry_period).min()
        df['don_exit_upper'] = df['high'].rolling(exit_period).max()
        df['don_exit_lower'] = df['low'].rolling(exit_period).min()
        
        adx_period = 14
        high, low, close = df['high'], df['low'], df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(span=adx_period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=adx_period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=adx_period, adjust=False).mean() / atr)
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        df['adx'] = dx.ewm(span=adx_period, adjust=False).mean()
        
        chop_period = 14
        high_low_range = high.rolling(chop_period).max() - low.rolling(chop_period).min()
        atr_sum = tr.rolling(chop_period).sum()
        df['chop'] = 100 * np.log10(atr_sum / (high_low_range + 1e-10)) / np.log10(chop_period)
        
        df['regime_ma'] = close.rolling(p['regime_ma_period']).mean()
        df['atr'] = tr.ewm(span=14, adjust=False).mean()
        
        returns = close.pct_change()
        df['realized_vol'] = returns.rolling(24).std() * np.sqrt(24 * 365)
        
        crash_mult = p.get('crash_vol_mult', 2.5)
        vol_ma = df['realized_vol'].rolling(168).mean()
        df['is_crash'] = df['realized_vol'] > (vol_ma * crash_mult)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        prepared_df = self.prepare_data(df)
        
        p = self.params
        n = len(prepared_df)
        signals = np.zeros(n)
        
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
        
        position = 0.0
        position_direction = 0
        bars_in_position = 0
        bars_since_exit = 9999
        entry_price = 0.0
        stop_price = 0.0
        
        adx_threshold = p['adx_gate_threshold']
        chop_threshold = p['chop_threshold']
        min_hold = p['min_hold_hours']
        cooldown = p['cooldown_hours']
        atr_stop = p['atr_stop_mult']
        vol_target = p['vol_target']
        leverage_cap = 1.0
        pos_change_threshold = p['position_change_threshold']
        
        for i in range(1, n):
            if np.isnan(don_entry_upper[i]) or np.isnan(adx_arr[i]):
                signals[i] = position
                continue
            
            if position_direction != 0:
                bars_in_position += 1
            else:
                bars_since_exit += 1
            
            if is_crash[i] and position_direction != 0:
                position_direction = 0
                position = 0.0
                bars_in_position = 0
                bars_since_exit = 0
                stop_price = 0.0
                signals[i] = 0.0
                continue
            
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
            
            gate_ok = True
            if is_crash[i]:
                gate_ok = False
            elif np.isnan(adx_arr[i]) or np.isnan(chop_arr[i]):
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
            
            if desired_direction == 0:
                desired_position = 0.0
            else:
                current_vol = max(realized_vol[i], 1e-6)
                vol_scalar = vol_target / current_vol
                base_size = np.clip(vol_scalar, 0.2, leverage_cap)
                desired_position = float(np.clip(desired_direction * base_size, -leverage_cap, leverage_cap))
            
            was_in_position = position_direction != 0
            
            if abs(desired_position - position) >= pos_change_threshold or desired_position == 0.0:
                position = desired_position
                position_direction = int(np.sign(position))
                
                if position_direction == 0:
                    bars_in_position = 0
                    stop_price = 0.0
                    if was_in_position:
                        bars_since_exit = 0
            
            signals[i] = position
        
        return pd.Series(signals, index=df.index)


def main():
    print("="*70)
    print("V2.5_1x OPTIMIZED - FULL TRADE DETAILS EXPORT")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    output_dir = PROJECT_ROOT / 'outputs_v2_5_1x_optimized'
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Load optimized params
    params_path = output_dir / 'best_params_v25_1x_optimized.json'
    with open(params_path) as f:
        params = json.load(f)
    
    print(f"\nParameters loaded from: {params_path.name}")
    print(f"  Leverage Cap: {params.get('leverage_cap', 1.0)}x (No Leverage)")
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_data(start_date="2020-01-01", end_date="2025-11-30", freq="1h")
    print(f"  Rows: {len(df):,}")
    print(f"  Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Generate signals
    print("\n[2/5] Generating signals...")
    strategy = Strategy1xOptimized(params)
    signals = strategy.generate_signals(df)
    print(f"  Signal range: [{signals.min():.2f}, {signals.max():.2f}]")
    
    # Run backtest
    print("\n[3/5] Running backtest...")
    bt_config = BacktestConfig(
        initial_capital=15000.0,
        commission_bps=4.0,
        slippage_bps=5.0,
        leverage=1.0,
        max_leverage=1.0
    )
    
    backtester = Backtester(bt_config)
    result = backtester.run(df.copy(), signals)
    
    print(f"  CAGR: {result.metrics['CAGR']:.1f}%")
    print(f"  MaxDD: {result.metrics['MaxDrawdown']:.1f}%")
    print(f"  Sharpe: {result.metrics['Sharpe']:.2f}")
    print(f"  Trades: {result.metrics['TradesCount']}")
    print(f"  Final Equity: ${result.metrics['FinalEquity']:,.2f}")
    
    # Process trades
    print("\n[4/5] Processing trades...")
    trades_df = result.trades.copy()
    
    if len(trades_df) > 0:
        trades_df.insert(0, 'trade_no', range(1, len(trades_df) + 1))
        trades_df['leverage_used'] = trades_df['size'] * trades_df['entry_price'] / bt_config.initial_capital
        trades_df['notional_value'] = trades_df['entry_price'] * trades_df['size']
        trades_df['pnl_pct'] = trades_df['return'] * 100
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['account_balance'] = bt_config.initial_capital + trades_df['cumulative_pnl']
        
        # Capital invested percentage
        trades_df['capital_invested_pct'] = (
            trades_df['notional_value'] / 
            trades_df['account_balance'].shift(1).fillna(bt_config.initial_capital)
        ) * 100
        
        # Win/loss streaks
        trades_df['pnl_positive'] = trades_df['pnl'] > 0
        trades_df['streak_id'] = (trades_df['pnl_positive'] != trades_df['pnl_positive'].shift()).cumsum()
        trades_df['streak_count'] = trades_df.groupby('streak_id').cumcount() + 1
        trades_df['streak_type'] = trades_df['pnl_positive'].apply(lambda x: 'W' if x else 'L')
        trades_df['streak'] = trades_df['streak_type'] + trades_df['streak_count'].astype(str)
        
        max_win_streak = trades_df[trades_df['pnl_positive']]['streak_count'].max()
        max_loss_streak = trades_df[~trades_df['pnl_positive']]['streak_count'].max()
        
        print(f"  Total Trades: {len(trades_df)}")
        print(f"  Max Win Streak: {max_win_streak}")
        print(f"  Max Loss Streak: {max_loss_streak}")
    else:
        max_win_streak = 0
        max_loss_streak = 0
    
    # Generate Excel report
    print("\n[5/5] Generating reports...")
    excel_path = output_dir / 'trade_report_v25_1x_optimized.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 1. Summary sheet
        summary_data = [
            {'metric': 'Strategy', 'value': 'V2.5_1x Optimized'},
            {'metric': 'Leverage', 'value': '1.0x (No Leverage)'},
            {'metric': 'Period', 'value': '2020-01-01 to 2025-11-30'},
            {'metric': 'Initial Capital', 'value': f"${bt_config.initial_capital:,.2f}"},
            {'metric': 'Final Equity', 'value': f"${result.metrics['FinalEquity']:,.2f}"},
            {'metric': 'Total Return', 'value': f"{result.metrics['TotalReturn']:.1f}%"},
            {'metric': 'CAGR', 'value': f"{result.metrics['CAGR']:.1f}%"},
            {'metric': 'MaxDrawdown', 'value': f"{result.metrics['MaxDrawdown']:.1f}%"},
            {'metric': 'Sharpe', 'value': f"{result.metrics['Sharpe']:.2f}"},
            {'metric': 'Sortino', 'value': f"{result.metrics.get('Sortino', 0):.2f}"},
            {'metric': 'Calmar', 'value': f"{result.metrics.get('Calmar', 0):.2f}"},
            {'metric': 'WinRate', 'value': f"{result.metrics['WinRate']:.1f}%"},
            {'metric': 'ProfitFactor', 'value': f"{result.metrics.get('ProfitFactor', 0):.2f}"},
            {'metric': 'TradesCount', 'value': result.metrics['TradesCount']},
            {'metric': 'AvgTradeReturn', 'value': f"{result.metrics.get('AvgTradeReturn', 0):.2f}%"},
            {'metric': 'Commission (bps)', 'value': bt_config.commission_bps},
            {'metric': 'Slippage (bps)', 'value': bt_config.slippage_bps},
            {'metric': 'MaxWinStreak', 'value': max_win_streak},
            {'metric': 'MaxLossStreak', 'value': max_loss_streak},
        ]
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # 2. Parameters sheet
        params_df = pd.DataFrame([{'Parameter': k, 'Value': v} for k, v in params.items()])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)
        
        # 3. All Trades sheet
        if len(trades_df) > 0:
            trade_cols = [
                'trade_no', 'entry_time', 'exit_time', 'side',
                'entry_price', 'exit_price', 'size', 'leverage_used',
                'notional_value', 'capital_invested_pct', 'pnl', 'pnl_pct', 
                'cumulative_pnl', 'account_balance', 'streak'
            ]
            existing_cols = [c for c in trade_cols if c in trades_df.columns]
            if 'holding_days' in trades_df.columns:
                existing_cols.append('holding_days')
            if 'fees' in trades_df.columns:
                existing_cols.append('fees')
            trades_df[existing_cols].to_excel(writer, sheet_name='All Trades', index=False)
        
        # 4. Monthly PnL
        equity = result.equity_curve
        if isinstance(equity, pd.DataFrame):
            equity = equity.iloc[:, 0]
        equity.index = pd.to_datetime(equity.index)
        equity_series = pd.Series(equity.values.astype(float), index=equity.index)
        
        monthly = equity_series.resample('M').last()
        monthly_values = monthly.values
        monthly_returns = (monthly_values[1:] / monthly_values[:-1] - 1) * 100
        
        monthly_df = pd.DataFrame({
            'Month': monthly.index[1:].strftime('%Y-%m'),
            'Return_Pct': monthly_returns,
            'Equity': monthly_values[1:]
        })
        monthly_df.to_excel(writer, sheet_name='Monthly PnL', index=False)
        
        # 5. Yearly Summary
        yearly = equity_series.resample('Y').last()
        yearly_values = yearly.values
        yearly_returns = (yearly_values[1:] / yearly_values[:-1] - 1) * 100
        
        yearly_df = pd.DataFrame({
            'Year': yearly.index[1:].year,
            'Return_Pct': yearly_returns,
            'Equity': yearly_values[1:]
        })
        yearly_df.to_excel(writer, sheet_name='Yearly Summary', index=False)
        
        # 6. Daily Equity
        daily = equity_series.resample('D').last().dropna()
        daily_df = pd.DataFrame({
            'Date': daily.index.strftime('%Y-%m-%d'),
            'Equity': daily.values
        })
        daily_df.to_excel(writer, sheet_name='Daily Equity', index=False)
    
    print(f"  Saved: {excel_path.name}")
    
    # Generate charts
    # 1. Equity Curve
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(equity_series.index, equity_series.values, 'g-', linewidth=1.5, label='V2.5_1x Optimized')
    ax.axhline(y=bt_config.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax.fill_between(equity_series.index, bt_config.initial_capital, equity_series.values, 
                    where=equity_series.values >= bt_config.initial_capital, alpha=0.3, color='green')
    ax.fill_between(equity_series.index, bt_config.initial_capital, equity_series.values, 
                    where=equity_series.values < bt_config.initial_capital, alpha=0.3, color='red')
    
    ax.set_ylabel('Equity ($)', fontsize=12)
    ax.set_title(f'V2.5_1x Optimized Equity Curve\nCAGR: {result.metrics["CAGR"]:.1f}% | MaxDD: {result.metrics["MaxDrawdown"]:.1f}% | Sharpe: {result.metrics["Sharpe"]:.2f}', 
                fontweight='bold', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: equity_curve.png")
    
    # 2. Drawdown Chart
    fig, ax = plt.subplots(figsize=(14, 5))
    
    eq_vals = equity_series.values.astype(float)
    running_max = np.maximum.accumulate(eq_vals)
    drawdown = (eq_vals - running_max) / running_max * 100
    
    ax.fill_between(equity_series.index, 0, drawdown, color='red', alpha=0.5)
    ax.axhline(y=-result.metrics['MaxDrawdown'], color='darkred', linestyle='--', 
               label=f'Max DD: {result.metrics["MaxDrawdown"]:.1f}%')
    
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('V2.5_1x Optimized Drawdown', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-40, 5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'drawdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: drawdown.png")
    
    # 3. Monthly Returns Heatmap
    if len(monthly_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 5))
        
        colors = ['red' if r < 0 else 'green' for r in monthly_df['Return_Pct']]
        bars = ax.bar(range(len(monthly_df)), monthly_df['Return_Pct'], color=colors, alpha=0.7)
        
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel('Monthly Return (%)', fontsize=12)
        ax.set_title('V2.5_1x Optimized Monthly Returns', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Show every 6th label
        tick_positions = range(0, len(monthly_df), 6)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([monthly_df['Month'].iloc[i] for i in tick_positions], rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'monthly_returns.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: monthly_returns.png")
    
    # 4. Trade Analysis
    if len(trades_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 4a. PnL Distribution
        ax = axes[0, 0]
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] <= 0]['pnl']
        ax.hist(wins, bins=30, color='green', alpha=0.7, label=f'Wins ({len(wins)})')
        ax.hist(losses, bins=30, color='red', alpha=0.7, label=f'Losses ({len(losses)})')
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('PnL ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Trade PnL Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4b. Cumulative PnL
        ax = axes[0, 1]
        ax.plot(trades_df['trade_no'], trades_df['cumulative_pnl'], 'g-', linewidth=1.5)
        ax.fill_between(trades_df['trade_no'], 0, trades_df['cumulative_pnl'],
                       where=trades_df['cumulative_pnl'] >= 0, alpha=0.3, color='green')
        ax.fill_between(trades_df['trade_no'], 0, trades_df['cumulative_pnl'],
                       where=trades_df['cumulative_pnl'] < 0, alpha=0.3, color='red')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Trade #')
        ax.set_ylabel('Cumulative PnL ($)')
        ax.set_title('Cumulative PnL by Trade', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 4c. Win Rate by Year
        ax = axes[1, 0]
        if 'entry_time' in trades_df.columns:
            trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
            yearly_wr = trades_df.groupby('year').apply(
                lambda x: (x['pnl'] > 0).sum() / len(x) * 100
            )
            bars = ax.bar(yearly_wr.index, yearly_wr.values, color='steelblue', alpha=0.8)
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Year')
            ax.set_ylabel('Win Rate (%)')
            ax.set_title('Win Rate by Year', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            for bar, val in zip(bars, yearly_wr.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{val:.0f}%', ha='center', fontsize=9)
        
        # 4d. Trade Size Distribution
        ax = axes[1, 1]
        ax.hist(trades_df['capital_invested_pct'], bins=30, color='purple', alpha=0.7, edgecolor='white')
        ax.axvline(x=100, color='red', linestyle='--', label='100% (Full Position)')
        ax.set_xlabel('Capital Invested (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Position Size Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'trade_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: trade_analysis.png")
    
    # 5. Price Chart with Trades
    if len(trades_df) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
        
        # Price with trades
        ax = axes[0]
        df_plot = df.copy()
        df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
        df_plot.set_index('timestamp', inplace=True)
        
        ax.plot(df_plot.index, df_plot['close'], 'gray', linewidth=0.5, alpha=0.7, label='ETH Price')
        
        # Mark entries and exits
        long_entries = trades_df[trades_df['side'] == 'long']
        short_entries = trades_df[trades_df['side'] == 'short']
        
        if len(long_entries) > 0:
            ax.scatter(pd.to_datetime(long_entries['entry_time']), long_entries['entry_price'],
                      marker='^', color='green', s=30, alpha=0.7, label='Long Entry')
        if len(short_entries) > 0:
            ax.scatter(pd.to_datetime(short_entries['entry_time']), short_entries['entry_price'],
                      marker='v', color='red', s=30, alpha=0.7, label='Short Entry')
        
        ax.set_ylabel('ETH Price ($)', fontsize=12)
        ax.set_title('ETH Price with Trade Entries (V2.5_1x Optimized)', fontweight='bold', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Signals
        ax2 = axes[1]
        signals_plot = signals.copy()
        signals_plot.index = pd.to_datetime(df['timestamp'])
        ax2.fill_between(signals_plot.index, 0, signals_plot.values, 
                        where=signals_plot.values > 0, alpha=0.5, color='green', label='Long')
        ax2.fill_between(signals_plot.index, 0, signals_plot.values, 
                        where=signals_plot.values < 0, alpha=0.5, color='red', label='Short')
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_ylabel('Position', fontsize=12)
        ax2.set_xlabel('Date')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1.2, 1.2)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'price_with_trades.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: price_with_trades.png")
    
    # Print final summary
    print("\n" + "="*70)
    print("V2.5_1x OPTIMIZED - REPORT COMPLETE")
    print("="*70)
    
    print(f"\n### Performance Summary ###")
    print(f"  CAGR: {result.metrics['CAGR']:.1f}%")
    print(f"  MaxDrawdown: {result.metrics['MaxDrawdown']:.1f}%")
    print(f"  Sharpe Ratio: {result.metrics['Sharpe']:.2f}")
    print(f"  Win Rate: {result.metrics['WinRate']:.1f}%")
    print(f"  Total Trades: {result.metrics['TradesCount']}")
    print(f"  Final Equity: ${result.metrics['FinalEquity']:,.2f}")
    
    print(f"\n### Streak Analysis ###")
    print(f"  Max Win Streak: {max_win_streak}")
    print(f"  Max Loss Streak: {max_loss_streak}")
    
    print(f"\n### Output Files ###")
    print(f"  - {excel_path}")
    for f in sorted(plots_dir.glob('*.png')):
        print(f"  - {f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
