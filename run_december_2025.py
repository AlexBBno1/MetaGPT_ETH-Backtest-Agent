"""
Run backtest for December 2025 (2025/12/1 - 2025/12/29)
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src.backtester import Backtester, BacktestConfig


class OptimizedRobustStrategyV2:
    """Strategy for generating signals."""
    
    def __init__(self, params: dict):
        self.params = params
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with indicators."""
        df = df.copy()
        p = self.params
        
        # Donchian Channels
        entry_period = p['donchian_entry_period']
        exit_period = p['donchian_exit_period']
        
        df['don_entry_upper'] = df['high'].rolling(entry_period).max()
        df['don_entry_lower'] = df['low'].rolling(entry_period).min()
        df['don_exit_upper'] = df['high'].rolling(exit_period).max()
        df['don_exit_lower'] = df['low'].rolling(exit_period).min()
        
        # ADX
        adx_period = 14
        high = df['high']
        low = df['low']
        close = df['close']
        
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
        
        # Choppiness Index
        chop_period = 14
        high_low_range = high.rolling(chop_period).max() - low.rolling(chop_period).min()
        atr_sum = tr.rolling(chop_period).sum()
        df['chop'] = 100 * np.log10(atr_sum / (high_low_range + 1e-10)) / np.log10(chop_period)
        
        # Regime MA
        df['regime_ma'] = close.rolling(p['regime_ma_period']).mean()
        
        # ATR for stops
        df['atr'] = tr.ewm(span=14, adjust=False).mean()
        
        # Realized volatility
        returns = close.pct_change()
        df['realized_vol'] = returns.rolling(24).std() * np.sqrt(24 * 365)
        
        # Crash detection
        crash_mult = p.get('crash_vol_mult', 2.5)
        vol_ma = df['realized_vol'].rolling(168).mean()
        df['is_crash'] = df['realized_vol'] > (vol_ma * crash_mult)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals."""
        prepared_df = self.prepare_data(df)
        
        p = self.params
        n = len(prepared_df)
        signals = np.zeros(n)
        
        # Pre-extract arrays
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
            if np.isnan(don_entry_upper[i]) or np.isnan(adx_arr[i]):
                signals[i] = position
                continue
            
            # Update counters
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
                # Entry check
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


def run_december_2025_backtest():
    """Run backtest specifically for December 2025."""
    print("="*70)
    print("ETH Strategy V2 - December 2025 Backtest")
    print("="*70)
    print(f"Period: 2025-12-01 to 2025-12-29")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load best parameters
    params_path = PROJECT_ROOT / 'outputs_v2' / 'best_params_v2.json'
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    print(f"\nStrategy Parameters:")
    print(f"  Leverage cap: {params.get('leverage_cap', 1.5):.2f}x")
    print(f"  ADX threshold: {params.get('adx_gate_threshold', 20)}")
    print(f"  Chop threshold: {params.get('chop_threshold', 60)}")
    
    # Load data - need extra history for indicator warmup
    print("\nLoading ETH data (with warmup period)...")
    df_full = load_data(
        start_date="2025-10-01",  # Extra months for indicator warmup
        end_date="2025-12-29",
        freq="1h"
    )
    print(f"  Full data: {len(df_full):,} rows")
    print(f"  Range: {df_full['timestamp'].min()} to {df_full['timestamp'].max()}")
    
    # Generate signals on full data (for indicator warmup)
    print("\nGenerating signals...")
    strategy = OptimizedRobustStrategyV2(params)
    signals_full = strategy.generate_signals(df_full)
    
    # Filter to December 2025 only
    dec_mask = df_full['timestamp'] >= '2025-12-01'
    df_dec = df_full[dec_mask].copy().reset_index(drop=True)
    signals_dec = signals_full[dec_mask].reset_index(drop=True)
    signals_dec.index = df_dec.index
    
    print(f"  December data: {len(df_dec):,} rows")
    print(f"  December range: {df_dec['timestamp'].min()} to {df_dec['timestamp'].max()}")
    print(f"  Non-zero signals: {(signals_dec != 0).sum():,}")
    
    # Run backtest for December
    print("\nRunning December 2025 backtest...")
    bt_config = BacktestConfig(
        initial_capital=15000.0,
        commission_bps=4.0,
        slippage_bps=5.0,
        leverage=params.get('leverage_cap', 1.5),
        max_leverage=params.get('leverage_cap', 1.5)
    )
    backtester = Backtester(bt_config)
    result = backtester.run(df_dec, signals_dec)
    
    # Process trades
    trades_df = result.trades.copy()
    
    if len(trades_df) > 0:
        trades_df['leverage_used'] = trades_df['size'] * trades_df['entry_price'] / bt_config.initial_capital
        trades_df['notional_value'] = trades_df['entry_price'] * trades_df['size']
        trades_df['pnl_pct'] = trades_df['return'] * 100
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['account_balance'] = bt_config.initial_capital + trades_df['cumulative_pnl']
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        trades_df.insert(0, 'trade_no', range(1, len(trades_df) + 1))
    
    # Print results
    print("\n" + "="*70)
    print("DECEMBER 2025 RESULTS")
    print("="*70)
    
    metrics = result.metrics
    print(f"\nPerformance Metrics:")
    print(f"  Initial Capital: $15,000")
    print(f"  Final Equity: ${metrics['FinalEquity']:,.2f}")
    print(f"  Total Return: {metrics['TotalReturn']:.2f}%")
    print(f"  Max Drawdown: {metrics['MaxDrawdown']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['Sharpe']:.2f}")
    print(f"  Win Rate: {metrics['WinRate']:.1f}%")
    
    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {len(trades_df)}")
    if len(trades_df) > 0:
        print(f"  Winning Trades: {(trades_df['pnl'] > 0).sum()}")
        print(f"  Losing Trades: {(trades_df['pnl'] < 0).sum()}")
        print(f"  Total P&L: ${trades_df['pnl'].sum():,.2f}")
        print(f"  Avg Trade P&L: ${trades_df['pnl'].mean():.2f}")
        print(f"  Best Trade: ${trades_df['pnl'].max():.2f}")
        print(f"  Worst Trade: ${trades_df['pnl'].min():.2f}")
        print(f"  Avg Leverage: {trades_df['leverage_used'].mean():.2f}x")
        print(f"  Avg Holding: {trades_df['holding_days'].mean():.2f} days")
    
    # Create output directory
    output_dir = PROJECT_ROOT / 'outputs_v2' / 'december_2025'
    output_dir.mkdir(exist_ok=True)
    
    # Save to Excel
    excel_path = output_dir / 'december_2025_report.xlsx'
    print(f"\nSaving Excel report: {excel_path}")
    
    # Daily P&L
    equity_df = result.equity_curve.copy()
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    equity_df.set_index('timestamp', inplace=True)
    
    daily_equity = equity_df['equity'].resample('D').last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    
    daily_pnl = pd.DataFrame({
        'date': daily_returns.index.strftime('%Y-%m-%d'),
        'equity': daily_equity.values[1:],
        'return_pct': daily_returns.values * 100,
        'pnl_usd': daily_equity.diff().values[1:],
    })
    
    # Metrics summary
    metrics_df = pd.DataFrame([
        {'metric': k, 'value': v} for k, v in metrics.items()
    ])
    
    # Parameters
    params_df = pd.DataFrame([
        {'parameter': k, 'value': v} for k, v in params.items()
    ])
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='Summary', index=False)
        params_df.to_excel(writer, sheet_name='Parameters', index=False)
        
        if len(trades_df) > 0:
            trade_cols = [
                'trade_no', 'entry_time', 'exit_time', 'side', 
                'entry_price', 'exit_price', 'size', 'leverage_used',
                'notional_value', 'pnl', 'pnl_pct', 'cumulative_pnl',
                'account_balance', 'holding_days', 'fees'
            ]
            trade_cols = [c for c in trade_cols if c in trades_df.columns]
            trades_df[trade_cols].to_excel(writer, sheet_name='All Trades', index=False)
        
        daily_pnl.to_excel(writer, sheet_name='Daily PnL', index=False)
    
    print("  [OK] Excel saved")
    
    # Create charts
    print("\nGenerating charts...")
    
    # 1. Equity curve
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    timestamps = pd.to_datetime(result.equity_curve['timestamp'])
    equity = result.equity_curve['equity']
    
    ax1 = axes[0]
    ax1.plot(timestamps, equity, 'b-', linewidth=1.5, label='Equity')
    ax1.fill_between(timestamps, 15000, equity, alpha=0.3, color='blue')
    ax1.axhline(y=15000, color='gray', linestyle='--', alpha=0.5, label='Initial $15K')
    
    if len(trades_df) > 0:
        for _, trade in trades_df.iterrows():
            color = 'green' if trade['side'] == 'long' else 'red'
            marker = '^' if trade['side'] == 'long' else 'v'
            ax1.axvline(x=trade['entry_time'], color=color, alpha=0.3, linewidth=0.5)
    
    ax1.set_title('December 2025 - Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity ($)')
    ax1.legend(loc='upper left')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # Position
    ax2 = axes[1]
    position = result.equity_curve['position']
    ax2.fill_between(timestamps, 0, position, where=position >= 0, color='green', alpha=0.6, label='Long')
    ax2.fill_between(timestamps, 0, position, where=position < 0, color='red', alpha=0.6, label='Short')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Position')
    ax2.legend(loc='upper right')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # Drawdown
    ax3 = axes[2]
    drawdown = result.equity_curve['drawdown'] * 100
    ax3.fill_between(timestamps, 0, drawdown, color='red', alpha=0.6)
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_xlabel('Date')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'december_2025_equity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: december_2025_equity.png")
    
    # 2. Price with trades
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1 = axes[0]
    ax1.plot(df_dec['timestamp'], df_dec['close'], 'gray', linewidth=1, alpha=0.8, label='ETH Price')
    
    if len(trades_df) > 0:
        for _, trade in trades_df.iterrows():
            color = 'green' if trade['side'] == 'long' else 'red'
            marker = '^' if trade['side'] == 'long' else 'v'
            ax1.scatter(trade['entry_time'], trade['entry_price'], marker=marker, 
                       color=color, s=100, zorder=5, edgecolor='black', linewidth=1)
            ax1.scatter(trade['exit_time'], trade['exit_price'], marker='x', 
                       color=color, s=80, zorder=5, linewidth=2)
            
            # Draw line between entry and exit
            ax1.plot([trade['entry_time'], trade['exit_time']], 
                    [trade['entry_price'], trade['exit_price']], 
                    color=color, alpha=0.5, linewidth=1, linestyle='--')
    
    ax1.set_title('December 2025 - ETH Price with Trades', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.legend(loc='upper left')
    
    # Position
    ax2 = axes[1]
    ax2.fill_between(timestamps, 0, position, where=position >= 0, color='green', alpha=0.6, label='Long')
    ax2.fill_between(timestamps, 0, position, where=position < 0, color='red', alpha=0.6, label='Short')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Position')
    ax2.set_xlabel('Date')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'december_2025_trades.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: december_2025_trades.png")
    
    # 3. Daily P&L bar chart
    if len(daily_pnl) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = ['green' if x >= 0 else 'red' for x in daily_pnl['pnl_usd']]
        bars = ax.bar(range(len(daily_pnl)), daily_pnl['pnl_usd'], color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xticks(range(len(daily_pnl)))
        ax.set_xticklabels(daily_pnl['date'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Daily P&L ($)')
        ax.set_title('December 2025 - Daily P&L', fontsize=14, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'december_2025_daily_pnl.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: december_2025_daily_pnl.png")
    
    # 4. Trade analysis (if trades exist)
    if len(trades_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # P&L distribution
        ax1 = axes[0, 0]
        trades_df['pnl'].hist(bins=15, ax=ax1, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax1.axvline(x=trades_df['pnl'].mean(), color='green', linestyle='--', linewidth=1, 
                   label=f'Mean: ${trades_df["pnl"].mean():.0f}')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Trade P&L Distribution', fontweight='bold')
        ax1.legend()
        
        # Cumulative P&L
        ax2 = axes[0, 1]
        ax2.plot(range(1, len(trades_df)+1), trades_df['cumulative_pnl'], 'b-', linewidth=1.5)
        ax2.fill_between(range(1, len(trades_df)+1), 0, trades_df['cumulative_pnl'], alpha=0.3)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative P&L ($)')
        ax2.set_title('Cumulative P&L', fontweight='bold')
        
        # Account balance
        ax3 = axes[1, 0]
        ax3.plot(range(1, len(trades_df)+1), trades_df['account_balance'], 'g-', linewidth=1.5)
        ax3.axhline(y=15000, color='gray', linestyle='--', alpha=0.5, label='Initial $15K')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Account Balance ($)')
        ax3.set_title('Account Balance', fontweight='bold')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax3.legend()
        
        # Win/Loss by side
        ax4 = axes[1, 1]
        side_stats = trades_df.groupby('side').agg({
            'pnl': ['count', 'sum', lambda x: (x > 0).sum()]
        })
        side_stats.columns = ['trades', 'total_pnl', 'wins']
        side_stats['win_rate'] = (side_stats['wins'] / side_stats['trades'] * 100).round(1)
        
        x = np.arange(len(side_stats))
        width = 0.35
        ax4.bar(x - width/2, side_stats['trades'], width, label='Total', color='steelblue', alpha=0.7)
        ax4.bar(x + width/2, side_stats['wins'], width, label='Wins', color='green', alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels(side_stats.index.str.upper())
        ax4.set_ylabel('Count')
        ax4.set_title('Trades by Side', fontweight='bold')
        ax4.legend()
        
        for i, (idx, row) in enumerate(side_stats.iterrows()):
            ax4.text(i, row['trades'] + 0.2, f"WR: {row['win_rate']:.0f}%", ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'december_2025_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: december_2025_analysis.png")
    
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - december_2025_report.xlsx")
    print(f"  - december_2025_equity.png")
    print(f"  - december_2025_trades.png")
    print(f"  - december_2025_daily_pnl.png")
    if len(trades_df) > 0:
        print(f"  - december_2025_analysis.png")
    
    # Print trade details
    if len(trades_df) > 0:
        print("\n" + "-"*70)
        print("TRADE DETAILS")
        print("-"*70)
        for _, trade in trades_df.iterrows():
            direction = "LONG" if trade['side'] == 'long' else "SHORT"
            pnl_sign = "+" if trade['pnl'] >= 0 else ""
            print(f"\n  Trade #{int(trade['trade_no'])} ({direction})")
            print(f"    Entry: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} @ ${trade['entry_price']:,.2f}")
            print(f"    Exit:  {trade['exit_time'].strftime('%Y-%m-%d %H:%M')} @ ${trade['exit_price']:,.2f}")
            print(f"    Size:  {trade['size']:.4f} ETH (Leverage: {trade['leverage_used']:.2f}x)")
            print(f"    P&L:   {pnl_sign}${trade['pnl']:.2f} ({pnl_sign}{trade['pnl_pct']:.2f}%)")
            print(f"    Balance: ${trade['account_balance']:,.2f}")
    
    return result, trades_df


if __name__ == "__main__":
    run_december_2025_backtest()


