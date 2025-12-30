"""
Export detailed trade data, monthly P&L, and charts to Excel.
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
    """Copy of strategy for generating signals."""
    
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
        """Generate trading signals with full trade tracking."""
        prepared_df = self.prepare_data(df)
        
        p = self.params
        n = len(prepared_df)
        signals = np.zeros(n)
        
        # Track trade details
        self.trade_details = []
        
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
        timestamps = df['timestamp'].values
        
        # State tracking
        position = 0.0
        position_direction = 0
        bars_in_position = 0
        bars_since_exit = 9999
        entry_price = 0.0
        stop_price = 0.0
        entry_idx = 0
        
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
            
            exit_reason = None
            
            # CRASH CHECK
            if is_crash[i] and position_direction != 0:
                exit_reason = 'crash_exit'
                self._record_exit(i, timestamps, close, position_direction, entry_price, entry_idx, exit_reason)
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
                    exit_reason = 'stop_loss'
                    self._record_exit(i, timestamps, close, position_direction, entry_price, entry_idx, exit_reason)
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
                    exit_reason = 'donchian_exit'
                elif position_direction == -1 and close[i] > don_exit_upper[i]:
                    exit_signal = True
                    exit_reason = 'donchian_exit'
                
                if exit_signal:
                    self._record_exit(i, timestamps, close, position_direction, entry_price, entry_idx, exit_reason)
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
                        entry_idx = i
                    elif close[i] < don_entry_lower[i-1] and close[i] < regime_ma[i]:
                        desired_direction = -1
                        entry_price = close[i]
                        stop_price = entry_price + atr_stop * atr_arr[i]
                        bars_in_position = 0
                        entry_idx = i
            
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
                elif not was_in_position and position_direction != 0:
                    # Record new entry
                    self._record_entry(i, timestamps, close, position_direction, position, 
                                       stop_price, adx_arr[i], chop_arr[i], realized_vol[i])
            
            signals[i] = position
        
        return pd.Series(signals, index=df.index)
    
    def _record_entry(self, idx, timestamps, close, direction, position_size, stop, adx, chop, vol):
        """Record entry for detailed trade log."""
        self.trade_details.append({
            'entry_idx': idx,
            'entry_time': pd.Timestamp(timestamps[idx]),
            'entry_price': close[idx],
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'position_size': abs(position_size),
            'leverage': abs(position_size),
            'stop_price': stop,
            'adx_at_entry': adx,
            'chop_at_entry': chop,
            'vol_at_entry': vol,
        })
    
    def _record_exit(self, idx, timestamps, close, direction, entry_price, entry_idx, reason):
        """Record exit for detailed trade log."""
        if self.trade_details and 'exit_time' not in self.trade_details[-1]:
            self.trade_details[-1]['exit_idx'] = idx
            self.trade_details[-1]['exit_time'] = pd.Timestamp(timestamps[idx])
            self.trade_details[-1]['exit_price'] = close[idx]
            self.trade_details[-1]['exit_reason'] = reason
            
            # Calculate P&L
            if direction == 1:  # Long
                pnl_pct = (close[idx] - entry_price) / entry_price
            else:  # Short
                pnl_pct = (entry_price - close[idx]) / entry_price
            
            self.trade_details[-1]['pnl_pct'] = pnl_pct * 100
            self.trade_details[-1]['bars_held'] = idx - entry_idx


def create_detailed_trade_excel(df, signals, params, output_path):
    """Create comprehensive Excel with trade details and monthly P&L."""
    print("\n" + "="*70)
    print("Generating Detailed Trade Report")
    print("="*70)
    
    # Run backtest
    bt_config = BacktestConfig(
        initial_capital=15000.0,
        commission_bps=4.0,
        slippage_bps=5.0,
        leverage=params.get('leverage_cap', 1.5),
        max_leverage=params.get('leverage_cap', 1.5)
    )
    backtester = Backtester(bt_config)
    result = backtester.run(df, signals)
    
    # Get strategy trade details
    strategy = OptimizedRobustStrategyV2(params)
    strategy.generate_signals(df)
    
    # Enhance trades with additional info
    trades_df = result.trades.copy()
    
    if len(trades_df) > 0:
        # Calculate additional fields
        trades_df['leverage_used'] = trades_df['size'] * trades_df['entry_price'] / bt_config.initial_capital
        trades_df['notional_value'] = trades_df['entry_price'] * trades_df['size']
        trades_df['pnl_pct'] = trades_df['return'] * 100
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        
        # Add account balance after each trade (starting from initial capital)
        trades_df['account_balance'] = bt_config.initial_capital + trades_df['cumulative_pnl']
        
        # Format for readability
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        # Add trade number
        trades_df.insert(0, 'trade_no', range(1, len(trades_df) + 1))
        
        print(f"  Total trades: {len(trades_df)}")
        print(f"  Win rate: {(trades_df['pnl'] > 0).mean() * 100:.1f}%")
        print(f"  Total P&L: ${trades_df['pnl'].sum():,.2f}")
    
    # Create monthly P&L summary
    equity_df = result.equity_curve.copy()
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    equity_df.set_index('timestamp', inplace=True)
    
    # Monthly returns
    monthly_equity = equity_df['equity'].resample('ME').last()
    monthly_returns = monthly_equity.pct_change().dropna()
    
    monthly_pnl = pd.DataFrame({
        'month': monthly_returns.index.strftime('%Y-%m'),
        'start_equity': monthly_equity.shift(1).values[1:],
        'end_equity': monthly_equity.values[1:],
        'return_pct': monthly_returns.values * 100,
        'pnl_usd': (monthly_equity.values[1:] - monthly_equity.shift(1).values[1:]),
    })
    monthly_pnl['cumulative_return_pct'] = ((monthly_equity.values[1:] / bt_config.initial_capital) - 1) * 100
    
    # Count trades per month
    if len(trades_df) > 0:
        trades_df['month'] = trades_df['entry_time'].dt.to_period('M').astype(str)
        trades_per_month = trades_df.groupby('month').agg({
            'trade_no': 'count',
            'pnl': 'sum'
        }).rename(columns={'trade_no': 'num_trades', 'pnl': 'trades_pnl'})
        
        monthly_pnl = monthly_pnl.merge(trades_per_month, left_on='month', right_index=True, how='left')
        monthly_pnl['num_trades'] = monthly_pnl['num_trades'].fillna(0).astype(int)
    
    print(f"  Monthly periods: {len(monthly_pnl)}")
    
    # Create yearly summary
    yearly_equity = equity_df['equity'].resample('YE').last()
    yearly_returns = yearly_equity.pct_change().dropna()
    
    yearly_summary = pd.DataFrame({
        'year': yearly_returns.index.year,
        'start_equity': yearly_equity.shift(1).values[1:],
        'end_equity': yearly_equity.values[1:],
        'return_pct': yearly_returns.values * 100,
        'pnl_usd': (yearly_equity.values[1:] - yearly_equity.shift(1).values[1:]),
    })
    
    # Parameters sheet
    params_df = pd.DataFrame([
        {'parameter': k, 'value': v} for k, v in params.items()
    ])
    
    # Metrics sheet
    metrics_df = pd.DataFrame([
        {'metric': k, 'value': v} for k, v in result.metrics.items()
    ])
    
    # Daily equity for charts
    daily_equity = equity_df['equity'].resample('D').last().dropna()
    daily_equity_df = pd.DataFrame({
        'date': daily_equity.index,
        'equity': daily_equity.values,
        'drawdown_pct': ((daily_equity - daily_equity.expanding().max()) / daily_equity.expanding().max() * 100).values
    })
    
    # Write to Excel
    print(f"\n  Writing to Excel: {output_path}")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary metrics
        metrics_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Parameters
        params_df.to_excel(writer, sheet_name='Parameters', index=False)
        
        # All trades
        if len(trades_df) > 0:
            # Reorder columns for better readability
            trade_cols = [
                'trade_no', 'entry_time', 'exit_time', 'side', 
                'entry_price', 'exit_price', 'size', 'leverage_used',
                'notional_value', 'pnl', 'pnl_pct', 'cumulative_pnl',
                'account_balance', 'holding_days', 'fees'
            ]
            trade_cols = [c for c in trade_cols if c in trades_df.columns]
            trades_export = trades_df[trade_cols].copy()
            trades_export.to_excel(writer, sheet_name='All Trades', index=False)
        
        # Monthly P&L
        monthly_pnl.to_excel(writer, sheet_name='Monthly PnL', index=False)
        
        # Yearly summary
        yearly_summary.to_excel(writer, sheet_name='Yearly Summary', index=False)
        
        # Daily equity
        daily_equity_df.to_excel(writer, sheet_name='Daily Equity', index=False)
    
    print("  [OK] Excel file created")
    return result, trades_df, monthly_pnl, daily_equity_df


def create_charts(df, result, trades_df, monthly_pnl, daily_equity_df, output_dir):
    """Create comprehensive charts."""
    print("\n  Generating charts...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Equity Curve with Trade Markers
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    equity_df = result.equity_curve
    timestamps = pd.to_datetime(equity_df['timestamp'])
    
    # Equity curve
    ax1 = axes[0]
    ax1.plot(timestamps, equity_df['equity'], 'b-', linewidth=1.5, label='Equity')
    ax1.fill_between(timestamps, 15000, equity_df['equity'], alpha=0.3, color='blue')
    ax1.axhline(y=15000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    
    # Mark trades
    if len(trades_df) > 0:
        longs = trades_df[trades_df['side'] == 'long']
        shorts = trades_df[trades_df['side'] == 'short']
        
        # Entry markers
        for _, trade in longs.iterrows():
            entry_eq = equity_df.loc[equity_df['timestamp'] >= trade['entry_time'], 'equity'].iloc[0] if len(equity_df[equity_df['timestamp'] >= trade['entry_time']]) > 0 else None
            if entry_eq is not None:
                ax1.scatter(trade['entry_time'], entry_eq, marker='^', color='green', s=30, alpha=0.7, zorder=5)
        
        for _, trade in shorts.iterrows():
            entry_eq = equity_df.loc[equity_df['timestamp'] >= trade['entry_time'], 'equity'].iloc[0] if len(equity_df[equity_df['timestamp'] >= trade['entry_time']]) > 0 else None
            if entry_eq is not None:
                ax1.scatter(trade['entry_time'], entry_eq, marker='v', color='red', s=30, alpha=0.7, zorder=5)
    
    ax1.set_title('Equity Curve with Trade Entries', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity ($)')
    ax1.legend(loc='upper left')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Position
    ax2 = axes[1]
    ax2.fill_between(timestamps, 0, equity_df['position'], 
                     where=equity_df['position'] >= 0, color='green', alpha=0.6, label='Long')
    ax2.fill_between(timestamps, 0, equity_df['position'], 
                     where=equity_df['position'] < 0, color='red', alpha=0.6, label='Short')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Position (Leverage)')
    ax2.legend(loc='upper right')
    ax2.set_ylim(-2, 2)
    
    # Drawdown
    ax3 = axes[2]
    drawdown = equity_df['drawdown'] * 100
    ax3.fill_between(timestamps, 0, drawdown, color='red', alpha=0.6)
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curve_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: equity_curve_detailed.png")
    
    # 2. Monthly P&L Bar Chart
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    ax1 = axes[0]
    colors = ['green' if x >= 0 else 'red' for x in monthly_pnl['return_pct']]
    bars = ax1.bar(range(len(monthly_pnl)), monthly_pnl['return_pct'], color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xticks(range(len(monthly_pnl)))
    ax1.set_xticklabels(monthly_pnl['month'], rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Monthly Return (%)')
    ax1.set_title('Monthly Returns', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, monthly_pnl['return_pct'])):
        ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                     xytext=(0, 3 if val >= 0 else -12), textcoords='offset points',
                     ha='center', fontsize=7)
    
    # Monthly P&L in USD
    ax2 = axes[1]
    colors = ['green' if x >= 0 else 'red' for x in monthly_pnl['pnl_usd']]
    bars = ax2.bar(range(len(monthly_pnl)), monthly_pnl['pnl_usd'], color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xticks(range(len(monthly_pnl)))
    ax2.set_xticklabels(monthly_pnl['month'], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Monthly P&L ($)')
    ax2.set_title('Monthly P&L (USD)', fontsize=14, fontweight='bold')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'monthly_pnl.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: monthly_pnl.png")
    
    # 3. Trade Analysis
    if len(trades_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # P&L Distribution
        ax1 = axes[0, 0]
        trades_df['pnl'].hist(bins=30, ax=ax1, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax1.axvline(x=trades_df['pnl'].mean(), color='green', linestyle='--', linewidth=1, label=f'Mean: ${trades_df["pnl"].mean():.0f}')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Trade P&L Distribution', fontweight='bold')
        ax1.legend()
        
        # Cumulative P&L
        ax2 = axes[0, 1]
        ax2.plot(range(1, len(trades_df)+1), trades_df['cumulative_pnl'], 'b-', linewidth=1.5)
        ax2.fill_between(range(1, len(trades_df)+1), 0, trades_df['cumulative_pnl'], alpha=0.3)
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative P&L ($)')
        ax2.set_title('Cumulative P&L by Trade', fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Leverage Distribution
        ax3 = axes[1, 0]
        trades_df['leverage_used'].hist(bins=20, ax=ax3, color='orange', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Leverage Used')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Leverage Distribution', fontweight='bold')
        
        # Win/Loss by side
        ax4 = axes[1, 1]
        side_stats = trades_df.groupby('side').agg({
            'pnl': ['count', 'sum', lambda x: (x > 0).sum()]
        }).round(2)
        side_stats.columns = ['trades', 'total_pnl', 'wins']
        side_stats['win_rate'] = (side_stats['wins'] / side_stats['trades'] * 100).round(1)
        
        x = np.arange(len(side_stats))
        width = 0.35
        ax4.bar(x - width/2, side_stats['trades'], width, label='Total Trades', color='steelblue', alpha=0.7)
        ax4.bar(x + width/2, side_stats['wins'], width, label='Winning Trades', color='green', alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels(side_stats.index.str.upper())
        ax4.set_ylabel('Count')
        ax4.set_title('Trades by Side', fontweight='bold')
        ax4.legend()
        
        # Add win rate text
        for i, (idx, row) in enumerate(side_stats.iterrows()):
            ax4.text(i, row['trades'] + 2, f"WR: {row['win_rate']:.0f}%", ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'trade_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: trade_analysis.png")
    
    # 4. Price with Entries/Exits
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1 = axes[0]
    ax1.plot(df['timestamp'], df['close'], 'gray', linewidth=0.8, alpha=0.8, label='ETH Price')
    
    if len(trades_df) > 0:
        for _, trade in trades_df.iterrows():
            color = 'green' if trade['side'] == 'long' else 'red'
            ax1.scatter(trade['entry_time'], trade['entry_price'], marker='^' if trade['side'] == 'long' else 'v', 
                       color=color, s=50, zorder=5, edgecolor='black', linewidth=0.5)
            ax1.scatter(trade['exit_time'], trade['exit_price'], marker='x', color=color, s=40, zorder=5)
    
    ax1.set_ylabel('ETH Price ($)')
    ax1.set_title('ETH Price with Trade Entries (▲/▼) and Exits (×)', fontsize=14, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.legend(loc='upper left')
    
    ax2 = axes[1]
    ax2.fill_between(timestamps, 0, equity_df['position'], 
                     where=equity_df['position'] >= 0, color='green', alpha=0.6, label='Long')
    ax2.fill_between(timestamps, 0, equity_df['position'], 
                     where=equity_df['position'] < 0, color='red', alpha=0.6, label='Short')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Position')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'price_with_trades.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: price_with_trades.png")
    
    print("  [OK] All charts created")


def main():
    print("="*70)
    print("ETH Strategy V2 - Detailed Trade Export")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load best parameters
    params_path = PROJECT_ROOT / 'outputs_v2' / 'best_params_v2.json'
    if not params_path.exists():
        print(f"Error: {params_path} not found. Run optimization first.")
        return
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    print(f"\nLoaded parameters from: {params_path}")
    print(f"  Leverage cap: {params.get('leverage_cap', 1.5):.2f}x")
    print(f"  ADX threshold: {params.get('adx_gate_threshold', 20)}")
    print(f"  Chop threshold: {params.get('chop_threshold', 60)}")
    
    # Load data
    print("\nLoading ETH data...")
    df = load_data(
        start_date="2020-01-01",
        end_date="2025-11-30",
        freq="1h"
    )
    print(f"  Rows: {len(df):,}")
    print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Generate signals
    print("\nGenerating signals...")
    strategy = OptimizedRobustStrategyV2(params)
    signals = strategy.generate_signals(df)
    print(f"  Signals generated: {(signals != 0).sum():,} non-zero")
    
    # Create output directory
    output_dir = PROJECT_ROOT / 'outputs_v2' / 'trade_details'
    output_dir.mkdir(exist_ok=True)
    
    # Create Excel report
    excel_path = output_dir / 'trade_report_v2.xlsx'
    result, trades_df, monthly_pnl, daily_equity_df = create_detailed_trade_excel(
        df, signals, params, excel_path
    )
    
    # Create charts
    create_charts(df, result, trades_df, monthly_pnl, daily_equity_df, output_dir)
    
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - trade_report_v2.xlsx (with sheets: Summary, Parameters, All Trades, Monthly PnL, Yearly Summary, Daily Equity)")
    print(f"  - equity_curve_detailed.png")
    print(f"  - monthly_pnl.png")
    print(f"  - trade_analysis.png")
    print(f"  - price_with_trades.png")
    
    # Print summary stats
    print(f"\n" + "-"*40)
    print("Quick Summary")
    print("-"*40)
    print(f"Total Trades: {len(trades_df)}")
    if len(trades_df) > 0:
        print(f"Win Rate: {(trades_df['pnl'] > 0).mean() * 100:.1f}%")
        print(f"Total P&L: ${trades_df['pnl'].sum():,.2f}")
        print(f"Avg Trade P&L: ${trades_df['pnl'].mean():.2f}")
        print(f"Avg Leverage: {trades_df['leverage_used'].mean():.2f}x")
        print(f"Avg Holding: {trades_df['holding_days'].mean():.1f} days")
    print(f"Final Equity: ${result.metrics['FinalEquity']:,.2f}")
    print(f"CAGR: {result.metrics['CAGR']:.2f}%")
    print(f"Max Drawdown: {result.metrics['MaxDrawdown']:.2f}%")
    print(f"Sharpe: {result.metrics['Sharpe']:.2f}")


if __name__ == "__main__":
    main()

