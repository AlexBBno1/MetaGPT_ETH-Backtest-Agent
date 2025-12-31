"""
V2.5_1x - No Leverage Version
Maximum position size limited to 100% of current capital (leverage_cap = 1.0)
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src.backtester import Backtester, BacktestConfig


class StrategyV25_1x:
    """V2.5 Strategy with 1x leverage cap (no leverage)."""
    
    def __init__(self, params: dict):
        self.params = params
        # Force leverage cap to 1.0
        self.params['leverage_cap'] = 1.0
    
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
        
        # Choppiness Index
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
        """Generate trading signals with 1x leverage cap."""
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
        leverage_cap = 1.0  # Force 1x leverage
        pos_change_threshold = p['position_change_threshold']
        
        for i in range(1, n):
            if np.isnan(don_entry_upper[i]) or np.isnan(adx_arr[i]):
                signals[i] = position
                continue
            
            if position_direction != 0:
                bars_in_position += 1
            else:
                bars_since_exit += 1
            
            # CRASH EXIT
            if is_crash[i] and position_direction != 0:
                position_direction = 0
                position = 0.0
                bars_in_position = 0
                bars_since_exit = 0
                stop_price = 0.0
                signals[i] = 0.0
                continue
            
            # STOP LOSS
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
            
            # DONCHIAN EXIT
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
            
            # REGIME GATE
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
                base_size = np.clip(vol_scalar, 0.2, leverage_cap)  # 1x max
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


def calculate_rolling_metrics(df: pd.DataFrame, signals: pd.Series, params: dict,
                              window_months: int = 12, step_months: int = 1) -> pd.DataFrame:
    """Calculate rolling window performance metrics."""
    
    bt_config = BacktestConfig(
        initial_capital=15000.0,
        commission_bps=4.0,
        slippage_bps=5.0,
        leverage=1.0,  # 1x
        max_leverage=1.0
    )
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()
    
    results = []
    current_start = start_date
    
    while True:
        window_end = current_start + pd.DateOffset(months=window_months)
        if window_end > end_date:
            break
        
        mask = (df['timestamp'] >= current_start) & (df['timestamp'] < window_end)
        window_df = df[mask].copy().reset_index(drop=True)
        window_signals = signals[mask].reset_index(drop=True)
        window_signals.index = window_df.index
        
        if len(window_df) > 100:
            try:
                backtester = Backtester(bt_config)
                result = backtester.run(window_df, window_signals)
                
                results.append({
                    'start': current_start,
                    'end': window_end,
                    'CAGR': result.metrics['CAGR'],
                    'MaxDD': result.metrics['MaxDrawdown'],
                    'Sharpe': result.metrics['Sharpe'],
                    'Trades': result.metrics['TradesCount'],
                })
            except:
                pass
        
        current_start += pd.DateOffset(months=step_months)
    
    return pd.DataFrame(results)


def main():
    print("="*70)
    print("V2.5_1x - NO LEVERAGE VERSION")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nKey Constraint: leverage_cap = 1.0 (100% of capital max per trade)")
    
    # Create output directory
    output_dir = PROJECT_ROOT / 'outputs_v2_5_1x'
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Load V2.5 parameters
    v25_params_path = PROJECT_ROOT / 'outputs_v2_5' / 'best_params_v25.json'
    with open(v25_params_path) as f:
        v25_params = json.load(f)
    
    print(f"\nOriginal V2.5 leverage_cap: {v25_params['leverage_cap']:.2f}x")
    print(f"V2.5_1x leverage_cap: 1.00x (forced)")
    
    # Create 1x params
    v25_1x_params = v25_params.copy()
    v25_1x_params['leverage_cap'] = 1.0
    
    # Save V2.5_1x params
    with open(output_dir / 'best_params_v25_1x.json', 'w') as f:
        json.dump(v25_1x_params, f, indent=2)
    
    # Load data
    print("\n[1/5] Loading ETH data...")
    df = load_data(start_date="2020-01-01", end_date="2025-11-30", freq="1h")
    print(f"  Rows: {len(df):,}")
    
    # Generate signals for both versions
    print("\n[2/5] Generating signals...")
    
    # V2.5 original
    strategy_v25 = StrategyV25_1x(v25_params.copy())
    strategy_v25.params['leverage_cap'] = v25_params['leverage_cap']  # Restore original
    signals_v25 = strategy_v25.generate_signals(df)
    
    # V2.5_1x
    strategy_v25_1x = StrategyV25_1x(v25_1x_params)
    signals_v25_1x = strategy_v25_1x.generate_signals(df)
    
    print(f"  V2.5 signal range: [{signals_v25.min():.2f}, {signals_v25.max():.2f}]")
    print(f"  V2.5_1x signal range: [{signals_v25_1x.min():.2f}, {signals_v25_1x.max():.2f}]")
    
    # Run backtests
    print("\n[3/5] Running backtests...")
    
    # V2.5 original config
    bt_config_v25 = BacktestConfig(
        initial_capital=15000.0,
        commission_bps=4.0,
        slippage_bps=5.0,
        leverage=v25_params['leverage_cap'],
        max_leverage=v25_params['leverage_cap']
    )
    
    # V2.5_1x config (1x leverage)
    bt_config_1x = BacktestConfig(
        initial_capital=15000.0,
        commission_bps=4.0,
        slippage_bps=5.0,
        leverage=1.0,
        max_leverage=1.0
    )
    
    backtester_v25 = Backtester(bt_config_v25)
    backtester_1x = Backtester(bt_config_1x)
    
    result_v25 = backtester_v25.run(df.copy(), signals_v25)
    result_1x = backtester_1x.run(df.copy(), signals_v25_1x)
    
    print(f"  V2.5:    CAGR={result_v25.metrics['CAGR']:.1f}%, Sharpe={result_v25.metrics['Sharpe']:.2f}, MaxDD={result_v25.metrics['MaxDrawdown']:.1f}%")
    print(f"  V2.5_1x: CAGR={result_1x.metrics['CAGR']:.1f}%, Sharpe={result_1x.metrics['Sharpe']:.2f}, MaxDD={result_1x.metrics['MaxDrawdown']:.1f}%")
    
    # Calculate rolling metrics
    print("\n[4/5] Calculating rolling metrics...")
    rolling_v25 = calculate_rolling_metrics(df, signals_v25, v25_params, 12, 1)
    rolling_1x = calculate_rolling_metrics(df, signals_v25_1x, v25_1x_params, 12, 1)
    
    # Process trades
    print("\n[5/5] Processing trades and generating reports...")
    
    trades_1x = result_1x.trades.copy()
    if len(trades_1x) > 0:
        trades_1x['trade_no'] = range(1, len(trades_1x) + 1)
        trades_1x['leverage_used'] = trades_1x['size'] * trades_1x['entry_price'] / bt_config_1x.initial_capital
        trades_1x['notional_value'] = trades_1x['entry_price'] * trades_1x['size']
        trades_1x['pnl_pct'] = trades_1x['return'] * 100
        trades_1x['cumulative_pnl'] = trades_1x['pnl'].cumsum()
        trades_1x['account_balance'] = bt_config_1x.initial_capital + trades_1x['cumulative_pnl']
        
        # Capital invested percentage
        trades_1x['capital_invested_pct'] = (
            trades_1x['notional_value'] / 
            trades_1x['account_balance'].shift(1).fillna(bt_config_1x.initial_capital)
        ) * 100
        
        # Win/loss streaks
        trades_1x['pnl_positive'] = trades_1x['pnl'] > 0
        trades_1x['streak_id'] = (trades_1x['pnl_positive'] != trades_1x['pnl_positive'].shift()).cumsum()
        trades_1x['streak_count'] = trades_1x.groupby('streak_id').cumcount() + 1
        trades_1x['streak_type'] = trades_1x['pnl_positive'].apply(lambda x: 'W' if x else 'L')
        trades_1x['streak'] = trades_1x['streak_type'] + trades_1x['streak_count'].astype(str)
        
        max_win_streak = trades_1x[trades_1x['pnl_positive']]['streak_count'].max() if not trades_1x[trades_1x['pnl_positive']].empty else 0
        max_loss_streak = trades_1x[~trades_1x['pnl_positive']]['streak_count'].max() if not trades_1x[~trades_1x['pnl_positive']].empty else 0
    else:
        max_win_streak = 0
        max_loss_streak = 0
    
    # Generate Excel report
    excel_path = output_dir / 'v25_1x_full_report.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 1. Summary comparison
        comparison_data = []
        metrics_to_compare = [
            ('CAGR', '%'), ('MaxDrawdown', '%'), ('Sharpe', ''), 
            ('TradesCount', ''), ('WinRate', '%'), ('FinalEquity', '$')
        ]
        
        for metric, unit in metrics_to_compare:
            v25_val = result_v25.metrics.get(metric, 0)
            v1x_val = result_1x.metrics.get(metric, 0)
            diff = v1x_val - v25_val
            
            comparison_data.append({
                'Metric': metric,
                'V2.5 (1.66x)': f"{v25_val:.2f}{unit}" if unit != '$' else f"${v25_val:,.2f}",
                'V2.5_1x (1.0x)': f"{v1x_val:.2f}{unit}" if unit != '$' else f"${v1x_val:,.2f}",
                'Diff': f"{diff:+.2f}{unit}" if unit != '$' else f"${diff:+,.2f}",
            })
        
        pd.DataFrame(comparison_data).to_excel(writer, sheet_name='V2.5 vs V2.5_1x Summary', index=False)
        
        # 2. Parameters
        params_df = pd.DataFrame([
            {'Parameter': k, 'V2.5': v25_params.get(k, 'N/A'), 'V2.5_1x': v25_1x_params.get(k, 'N/A')}
            for k in v25_1x_params.keys()
        ])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)
        
        # 3. Rolling Performance
        if len(rolling_v25) > 0 and len(rolling_1x) > 0:
            min_len = min(len(rolling_v25), len(rolling_1x))
            rolling_comparison = pd.DataFrame({
                'Window': range(1, min_len + 1),
                'V25_CAGR': rolling_v25['CAGR'].values[:min_len],
                'V25_1x_CAGR': rolling_1x['CAGR'].values[:min_len],
                'V25_MaxDD': rolling_v25['MaxDD'].values[:min_len],
                'V25_1x_MaxDD': rolling_1x['MaxDD'].values[:min_len],
            })
            rolling_comparison.to_excel(writer, sheet_name='Rolling 12M', index=False)
        
        # 4. All Trades
        if len(trades_1x) > 0:
            trade_cols = [c for c in [
                'trade_no', 'entry_time', 'exit_time', 'side',
                'entry_price', 'exit_price', 'size', 'leverage_used',
                'notional_value', 'capital_invested_pct', 'pnl', 'pnl_pct', 
                'cumulative_pnl', 'account_balance', 'streak',
                'holding_days', 'fees'
            ] if c in trades_1x.columns]
            trades_1x[trade_cols].to_excel(writer, sheet_name='All Trades', index=False)
        
        # 5. Monthly Performance
        equity_1x = result_1x.equity_curve.copy()
        if isinstance(equity_1x, pd.DataFrame):
            equity_1x = equity_1x.iloc[:, 0]
        equity_1x.index = pd.to_datetime(equity_1x.index)
        equity_series = pd.Series(equity_1x.values.astype(float), index=equity_1x.index)
        monthly = equity_series.resample('M').last()
        monthly_values = monthly.values
        monthly_returns_values = (monthly_values[1:] / monthly_values[:-1] - 1) * 100
        
        monthly_df = pd.DataFrame({
            'Month': monthly.index[1:].strftime('%Y-%m'),
            'Return_Pct': monthly_returns_values,
            'Equity': monthly_values[1:]
        })
        monthly_df.to_excel(writer, sheet_name='Monthly Returns', index=False)
        
        # 6. Streak Summary
        streak_summary = pd.DataFrame([
            {'Metric': 'Max Win Streak', 'Value': max_win_streak},
            {'Metric': 'Max Loss Streak', 'Value': max_loss_streak},
            {'Metric': 'Total Trades', 'Value': len(trades_1x)},
            {'Metric': 'Win Rate', 'Value': f"{result_1x.metrics['WinRate']:.1f}%"},
            {'Metric': 'Leverage Cap', 'Value': '1.0x (No Leverage)'},
        ])
        streak_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"  Saved: {excel_path.name}")
    
    # Generate charts
    # 1. Equity Curve Comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    
    equity_v25 = result_v25.equity_curve
    equity_1x_plot = result_1x.equity_curve
    
    if isinstance(equity_v25, pd.DataFrame):
        equity_v25 = equity_v25.iloc[:, 0]
    if isinstance(equity_1x_plot, pd.DataFrame):
        equity_1x_plot = equity_1x_plot.iloc[:, 0]
    
    ax.plot(equity_v25.index, equity_v25.values, 'b-', linewidth=1.5, 
            label=f'V2.5 (1.66x): CAGR={result_v25.metrics["CAGR"]:.1f}%', alpha=0.8)
    ax.plot(equity_1x_plot.index, equity_1x_plot.values, 'g-', linewidth=1.5, 
            label=f'V2.5_1x (1.0x): CAGR={result_1x.metrics["CAGR"]:.1f}%', alpha=0.8)
    ax.axhline(y=bt_config_1x.initial_capital, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_ylabel('Equity ($)')
    ax.set_title('V2.5 vs V2.5_1x Equity Curve Comparison\n(With Leverage vs No Leverage)', fontweight='bold', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'equity_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: equity_comparison.png")
    
    # 2. Drawdown Comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    def calculate_drawdown(equity):
        if isinstance(equity, pd.DataFrame):
            equity = equity.iloc[:, 0]
        eq_values = equity.values.astype(float)
        running_max = np.maximum.accumulate(eq_values)
        drawdown = (eq_values - running_max) / running_max * 100
        return pd.Series(drawdown, index=equity.index)
    
    dd_v25 = calculate_drawdown(result_v25.equity_curve)
    dd_1x = calculate_drawdown(result_1x.equity_curve)
    
    axes[0].fill_between(dd_v25.index, 0, dd_v25.values, color='blue', alpha=0.5, label='V2.5 (1.66x)')
    axes[0].set_ylabel('Drawdown (%)')
    axes[0].set_title(f'V2.5 Drawdown (MaxDD: {result_v25.metrics["MaxDrawdown"]:.1f}%)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-50, 5)
    
    axes[1].fill_between(dd_1x.index, 0, dd_1x.values, color='green', alpha=0.5, label='V2.5_1x (1.0x)')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_title(f'V2.5_1x Drawdown (MaxDD: {result_1x.metrics["MaxDrawdown"]:.1f}%)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-50, 5)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'drawdown_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: drawdown_comparison.png")
    
    # 3. Rolling CAGR comparison
    if len(rolling_v25) > 0 and len(rolling_1x) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        min_len = min(len(rolling_v25), len(rolling_1x))
        ax.plot(range(min_len), rolling_v25['CAGR'].values[:min_len], 'b-', linewidth=1.5, label='V2.5 (1.66x)', alpha=0.8)
        ax.plot(range(min_len), rolling_1x['CAGR'].values[:min_len], 'g-', linewidth=1.5, label='V2.5_1x (1.0x)', alpha=0.8)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        ax.set_ylabel('Rolling 12M CAGR (%)')
        ax.set_xlabel('Rolling Window')
        ax.set_title('Rolling 12-Month CAGR Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'rolling_cagr.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: rolling_cagr.png")
    
    # Print final report
    print("\n" + "="*70)
    print("V2.5 vs V2.5_1x FINAL COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'V2.5 (1.66x)':>15} {'V2.5_1x (1.0x)':>15} {'Change':>15}")
    print("-"*70)
    print(f"{'CAGR':<25} {result_v25.metrics['CAGR']:>14.1f}% {result_1x.metrics['CAGR']:>14.1f}% {result_1x.metrics['CAGR']-result_v25.metrics['CAGR']:>+14.1f}%")
    print(f"{'MaxDrawdown':<25} {result_v25.metrics['MaxDrawdown']:>14.1f}% {result_1x.metrics['MaxDrawdown']:>14.1f}% {result_1x.metrics['MaxDrawdown']-result_v25.metrics['MaxDrawdown']:>+14.1f}%")
    print(f"{'Sharpe Ratio':<25} {result_v25.metrics['Sharpe']:>15.2f} {result_1x.metrics['Sharpe']:>15.2f} {result_1x.metrics['Sharpe']-result_v25.metrics['Sharpe']:>+15.2f}")
    print(f"{'Final Equity':<25} ${result_v25.metrics['FinalEquity']:>13,.0f} ${result_1x.metrics['FinalEquity']:>13,.0f} ${result_1x.metrics['FinalEquity']-result_v25.metrics['FinalEquity']:>+13,.0f}")
    print(f"{'Trades':<25} {result_v25.metrics['TradesCount']:>15.0f} {result_1x.metrics['TradesCount']:>15.0f}")
    print(f"{'Win Rate':<25} {result_v25.metrics['WinRate']:>14.1f}% {result_1x.metrics['WinRate']:>14.1f}%")
    
    if len(rolling_v25) > 0 and len(rolling_1x) > 0:
        print(f"\n### Rolling 12M Stability ###")
        print(f"{'Median CAGR':<25} {rolling_v25['CAGR'].median():>14.1f}% {rolling_1x['CAGR'].median():>14.1f}%")
        print(f"{'Min CAGR':<25} {rolling_v25['CAGR'].min():>14.1f}% {rolling_1x['CAGR'].min():>14.1f}%")
        print(f"{'CAGR Std Dev':<25} {rolling_v25['CAGR'].std():>14.1f}% {rolling_1x['CAGR'].std():>14.1f}%")
    
    print(f"\n### Trade Analysis (V2.5_1x) ###")
    print(f"  Max Win Streak: {max_win_streak}")
    print(f"  Max Loss Streak: {max_loss_streak}")
    print(f"  Max Capital Invested: {trades_1x['capital_invested_pct'].max():.1f}% (capped at 100%)" if len(trades_1x) > 0 else "  No trades")
    
    print(f"\n### Key Insight ###")
    leverage_reduction_pct = (1 - 1.0/v25_params['leverage_cap']) * 100
    cagr_reduction_pct = (result_v25.metrics['CAGR'] - result_1x.metrics['CAGR']) / result_v25.metrics['CAGR'] * 100
    dd_reduction_pct = (result_v25.metrics['MaxDrawdown'] - result_1x.metrics['MaxDrawdown']) / result_v25.metrics['MaxDrawdown'] * 100
    
    print(f"  Leverage reduced by: {leverage_reduction_pct:.1f}% (1.66x -> 1.0x)")
    print(f"  CAGR reduced by: {cagr_reduction_pct:.1f}%")
    print(f"  MaxDD reduced by: {dd_reduction_pct:.1f}%")
    print(f"  Risk-adjusted: {'Better' if result_1x.metrics['Sharpe'] >= result_v25.metrics['Sharpe'] else 'Similar'}")
    
    print(f"\n### OUTPUT FILES ###")
    print(f"  - {output_dir / 'best_params_v25_1x.json'}")
    print(f"  - {excel_path}")
    for f in plots_dir.glob('*.png'):
        print(f"  - {f}")
    
    print("\n" + "="*70)
    print("V2.5_1x REPORT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
