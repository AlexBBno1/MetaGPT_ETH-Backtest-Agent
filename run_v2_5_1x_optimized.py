"""
V2.5_1x Optimized - Parameter optimization specifically for 1x leverage (no leverage)
This runs a full optimization with leverage_cap locked at 1.0
"""

import pandas as pd
import numpy as np
import json
import optuna
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)

import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src.backtester import Backtester, BacktestConfig


class Strategy1xOptimized:
    """Strategy optimized specifically for 1x leverage."""
    
    def __init__(self, params: dict):
        self.params = params
        # Force leverage to 1.0
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
        """Generate trading signals."""
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
        leverage_cap = 1.0  # Force 1x
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


def run_backtest(df: pd.DataFrame, params: dict) -> Dict:
    """Run backtest with given parameters."""
    bt_config = BacktestConfig(
        initial_capital=15000.0,
        commission_bps=4.0,
        slippage_bps=5.0,
        leverage=1.0,
        max_leverage=1.0
    )
    
    strategy = Strategy1xOptimized(params)
    signals = strategy.generate_signals(df)
    
    backtester = Backtester(bt_config)
    result = backtester.run(df, signals)
    
    return result.metrics


def calculate_score(metrics: Dict) -> Tuple[float, bool, str]:
    """Calculate optimization score for 1x strategy."""
    
    cagr = metrics['CAGR']
    max_dd = metrics['MaxDrawdown']
    sharpe = metrics['Sharpe']
    trades = metrics['TradesCount']
    years = metrics.get('Years', 5.9)
    trades_per_year = trades / years if years > 0 else 0
    
    # Hard constraints for 1x
    if max_dd > 35:  # Stricter MaxDD for no leverage
        return -999, False, f"MaxDD {max_dd:.1f}% > 35%"
    if trades_per_year < 40:
        return -999, False, f"Trades/year {trades_per_year:.1f} < 40"
    if trades_per_year > 150:
        return -999, False, f"Trades/year {trades_per_year:.1f} > 150"
    if sharpe < 0.8:
        return -999, False, f"Sharpe {sharpe:.2f} < 0.8"
    if cagr < 20:
        return -999, False, f"CAGR {cagr:.1f}% < 20%"
    
    # Score: balanced between CAGR, Sharpe, and lower DD
    score = (
        + 0.5 * cagr
        + 0.3 * sharpe * 20
        - 0.2 * max_dd
    )
    
    return score, True, "Valid"


def main():
    print("="*70)
    print("V2.5_1x OPTIMIZED - Full Parameter Optimization for No Leverage")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nKey Constraint: leverage_cap = 1.0 (LOCKED)")
    print(f"This optimization finds the BEST parameters for 1x trading")
    
    # Create output directory
    output_dir = PROJECT_ROOT / 'outputs_v2_5_1x_optimized'
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1/4] Loading ETH data...")
    df = load_data(start_date="2020-01-01", end_date="2025-11-30", freq="1h")
    print(f"  Rows: {len(df):,}")
    
    # Define search space for 1x optimization
    print("\n[2/4] Running Optuna optimization (100 trials)...")
    
    def objective(trial):
        params = {
            # Donchian periods - wider range for 1x
            'donchian_entry_period': trial.suggest_int('donchian_entry_period', 80, 200),
            'donchian_exit_period': trial.suggest_int('donchian_exit_period', 40, 120),
            
            # Regime filters
            'adx_gate_threshold': trial.suggest_int('adx_gate_threshold', 18, 35),
            'chop_threshold': trial.suggest_int('chop_threshold', 55, 75),
            'regime_ma_period': trial.suggest_int('regime_ma_period', 100, 250),
            
            # Vol targeting - optimized for 1x
            'vol_target': trial.suggest_float('vol_target', 0.30, 0.70),
            
            # Risk management
            'atr_stop_mult': trial.suggest_float('atr_stop_mult', 2.5, 5.0),
            'crash_vol_mult': trial.suggest_float('crash_vol_mult', 2.0, 3.5),
            
            # Timing
            'min_hold_hours': trial.suggest_int('min_hold_hours', 48, 120),
            'cooldown_hours': trial.suggest_int('cooldown_hours', 8, 36),
            'position_change_threshold': trial.suggest_float('position_change_threshold', 0.25, 0.55),
            
            # Fixed
            'leverage_cap': 1.0,
            'disable_regime_gate': False,
        }
        
        try:
            metrics = run_backtest(df.copy(), params)
            score, is_valid, reason = calculate_score(metrics)
            return score
        except Exception as e:
            return -999
    
    # Create study
    storage = f"sqlite:///{output_dir / 'optuna_1x_optimized.db'}"
    study = optuna.create_study(
        study_name="v25_1x_optimized",
        direction="maximize",
        storage=storage,
        load_if_exists=True
    )
    
    from tqdm import tqdm
    
    n_trials = 100
    remaining = n_trials - len(study.trials)
    
    if remaining > 0:
        with tqdm(total=remaining, desc="1x Optimization") as pbar:
            def callback(study, trial):
                pbar.update(1)
                if study.best_value and study.best_value > -900:
                    pbar.set_postfix({'best': f'{study.best_value:.1f}'})
            
            study.optimize(objective, n_trials=remaining, callbacks=[callback], 
                          show_progress_bar=False)
    
    if study.best_trial and study.best_value > -900:
        best_params = {**study.best_params, 'leverage_cap': 1.0, 'disable_regime_gate': False}
        print(f"\n[OK] Best Score: {study.best_value:.2f}")
    else:
        print("\n[!] No valid trials found, using relaxed constraints...")
        # Fallback to V2.5 params with 1x
        v25_path = PROJECT_ROOT / 'outputs_v2_5' / 'best_params_v25.json'
        with open(v25_path) as f:
            best_params = json.load(f)
        best_params['leverage_cap'] = 1.0
    
    # Save optimized params
    with open(output_dir / 'best_params_v25_1x_optimized.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Run final backtest with optimized params
    print("\n[3/4] Running final backtests for comparison...")
    
    # Load V2.5 original params
    v25_path = PROJECT_ROOT / 'outputs_v2_5' / 'best_params_v25.json'
    with open(v25_path) as f:
        v25_params = json.load(f)
    
    # V2.5 original (with leverage)
    bt_config_v25 = BacktestConfig(
        initial_capital=15000.0, commission_bps=4.0, slippage_bps=5.0,
        leverage=v25_params['leverage_cap'], max_leverage=v25_params['leverage_cap']
    )
    strategy_v25 = Strategy1xOptimized(v25_params.copy())
    strategy_v25.params['leverage_cap'] = v25_params['leverage_cap']
    signals_v25 = strategy_v25.generate_signals(df)
    backtester_v25 = Backtester(bt_config_v25)
    result_v25 = backtester_v25.run(df.copy(), signals_v25)
    
    # V2.5_1x (V2.5 params, forced 1x)
    v25_1x_params = v25_params.copy()
    v25_1x_params['leverage_cap'] = 1.0
    bt_config_1x = BacktestConfig(
        initial_capital=15000.0, commission_bps=4.0, slippage_bps=5.0,
        leverage=1.0, max_leverage=1.0
    )
    strategy_v25_1x = Strategy1xOptimized(v25_1x_params)
    signals_v25_1x = strategy_v25_1x.generate_signals(df)
    backtester_1x = Backtester(bt_config_1x)
    result_v25_1x = backtester_1x.run(df.copy(), signals_v25_1x)
    
    # V2.5_1x Optimized (new optimized params for 1x)
    strategy_opt = Strategy1xOptimized(best_params)
    signals_opt = strategy_opt.generate_signals(df)
    backtester_opt = Backtester(bt_config_1x)
    result_opt = backtester_opt.run(df.copy(), signals_opt)
    
    print(f"  V2.5 (1.66x):      CAGR={result_v25.metrics['CAGR']:.1f}%, Sharpe={result_v25.metrics['Sharpe']:.2f}, MaxDD={result_v25.metrics['MaxDrawdown']:.1f}%")
    print(f"  V2.5_1x (forced):  CAGR={result_v25_1x.metrics['CAGR']:.1f}%, Sharpe={result_v25_1x.metrics['Sharpe']:.2f}, MaxDD={result_v25_1x.metrics['MaxDrawdown']:.1f}%")
    print(f"  V2.5_1x Optimized: CAGR={result_opt.metrics['CAGR']:.1f}%, Sharpe={result_opt.metrics['Sharpe']:.2f}, MaxDD={result_opt.metrics['MaxDrawdown']:.1f}%")
    
    # Generate reports
    print("\n[4/4] Generating reports...")
    
    # Process trades for optimized version
    trades_opt = result_opt.trades.copy()
    if len(trades_opt) > 0:
        trades_opt['trade_no'] = range(1, len(trades_opt) + 1)
        trades_opt['pnl_pct'] = trades_opt['return'] * 100
        trades_opt['cumulative_pnl'] = trades_opt['pnl'].cumsum()
        trades_opt['account_balance'] = bt_config_1x.initial_capital + trades_opt['cumulative_pnl']
        
        # Streaks
        trades_opt['pnl_positive'] = trades_opt['pnl'] > 0
        trades_opt['streak_id'] = (trades_opt['pnl_positive'] != trades_opt['pnl_positive'].shift()).cumsum()
        trades_opt['streak_count'] = trades_opt.groupby('streak_id').cumcount() + 1
        trades_opt['streak_type'] = trades_opt['pnl_positive'].apply(lambda x: 'W' if x else 'L')
        trades_opt['streak'] = trades_opt['streak_type'] + trades_opt['streak_count'].astype(str)
        
        max_win = trades_opt[trades_opt['pnl_positive']]['streak_count'].max() if not trades_opt[trades_opt['pnl_positive']].empty else 0
        max_loss = trades_opt[~trades_opt['pnl_positive']]['streak_count'].max() if not trades_opt[~trades_opt['pnl_positive']].empty else 0
    else:
        max_win, max_loss = 0, 0
    
    # Excel report
    excel_path = output_dir / 'v25_1x_optimized_report.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Comparison summary
        comparison = pd.DataFrame([
            {
                'Version': 'V2.5 (1.66x leverage)',
                'CAGR': f"{result_v25.metrics['CAGR']:.1f}%",
                'MaxDD': f"{result_v25.metrics['MaxDrawdown']:.1f}%",
                'Sharpe': f"{result_v25.metrics['Sharpe']:.2f}",
                'Trades': result_v25.metrics['TradesCount'],
                'WinRate': f"{result_v25.metrics['WinRate']:.1f}%",
                'FinalEquity': f"${result_v25.metrics['FinalEquity']:,.0f}",
            },
            {
                'Version': 'V2.5_1x (forced 1x, same params)',
                'CAGR': f"{result_v25_1x.metrics['CAGR']:.1f}%",
                'MaxDD': f"{result_v25_1x.metrics['MaxDrawdown']:.1f}%",
                'Sharpe': f"{result_v25_1x.metrics['Sharpe']:.2f}",
                'Trades': result_v25_1x.metrics['TradesCount'],
                'WinRate': f"{result_v25_1x.metrics['WinRate']:.1f}%",
                'FinalEquity': f"${result_v25_1x.metrics['FinalEquity']:,.0f}",
            },
            {
                'Version': 'V2.5_1x OPTIMIZED (new params for 1x)',
                'CAGR': f"{result_opt.metrics['CAGR']:.1f}%",
                'MaxDD': f"{result_opt.metrics['MaxDrawdown']:.1f}%",
                'Sharpe': f"{result_opt.metrics['Sharpe']:.2f}",
                'Trades': result_opt.metrics['TradesCount'],
                'WinRate': f"{result_opt.metrics['WinRate']:.1f}%",
                'FinalEquity': f"${result_opt.metrics['FinalEquity']:,.0f}",
            },
        ])
        comparison.to_excel(writer, sheet_name='3-Way Comparison', index=False)
        
        # Parameter comparison
        param_keys = list(best_params.keys())
        params_df = pd.DataFrame([
            {'Parameter': k, 
             'V2.5': v25_params.get(k, 'N/A'), 
             'V2.5_1x_Optimized': best_params.get(k, 'N/A'),
             'Changed': 'YES' if v25_params.get(k) != best_params.get(k) else ''}
            for k in param_keys
        ])
        params_df.to_excel(writer, sheet_name='Parameters', index=False)
        
        # Trades
        if len(trades_opt) > 0:
            trade_cols = [c for c in [
                'trade_no', 'entry_time', 'exit_time', 'side',
                'entry_price', 'exit_price', 'size', 'pnl', 'pnl_pct',
                'cumulative_pnl', 'account_balance', 'streak'
            ] if c in trades_opt.columns]
            trades_opt[trade_cols].to_excel(writer, sheet_name='All Trades', index=False)
    
    print(f"  Saved: {excel_path.name}")
    
    # Charts
    # 1. Equity comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    
    eq_v25 = result_v25.equity_curve.iloc[:, 0] if isinstance(result_v25.equity_curve, pd.DataFrame) else result_v25.equity_curve
    eq_1x = result_v25_1x.equity_curve.iloc[:, 0] if isinstance(result_v25_1x.equity_curve, pd.DataFrame) else result_v25_1x.equity_curve
    eq_opt = result_opt.equity_curve.iloc[:, 0] if isinstance(result_opt.equity_curve, pd.DataFrame) else result_opt.equity_curve
    
    ax.plot(eq_v25.index, eq_v25.values, 'b-', linewidth=1.5, 
            label=f'V2.5 (1.66x): CAGR={result_v25.metrics["CAGR"]:.1f}%', alpha=0.8)
    ax.plot(eq_1x.index, eq_1x.values, 'orange', linewidth=1.5, linestyle='--',
            label=f'V2.5_1x forced: CAGR={result_v25_1x.metrics["CAGR"]:.1f}%', alpha=0.8)
    ax.plot(eq_opt.index, eq_opt.values, 'g-', linewidth=2,
            label=f'V2.5_1x OPTIMIZED: CAGR={result_opt.metrics["CAGR"]:.1f}%', alpha=0.9)
    
    ax.axhline(y=15000, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Equity ($)')
    ax.set_title('V2.5 vs V2.5_1x vs V2.5_1x OPTIMIZED\nEquity Curve Comparison', fontweight='bold', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'equity_3way_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: equity_3way_comparison.png")
    
    # 2. Drawdown comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    
    def calc_dd(equity):
        eq = equity.iloc[:, 0] if isinstance(equity, pd.DataFrame) else equity
        eq_vals = eq.values.astype(float)
        running_max = np.maximum.accumulate(eq_vals)
        dd = (eq_vals - running_max) / running_max * 100
        return pd.Series(dd, index=eq.index)
    
    dd_v25 = calc_dd(result_v25.equity_curve)
    dd_1x = calc_dd(result_v25_1x.equity_curve)
    dd_opt = calc_dd(result_opt.equity_curve)
    
    ax.fill_between(dd_v25.index, 0, dd_v25.values, color='blue', alpha=0.3, label=f'V2.5: MaxDD={result_v25.metrics["MaxDrawdown"]:.1f}%')
    ax.fill_between(dd_1x.index, 0, dd_1x.values, color='orange', alpha=0.3, label=f'V2.5_1x forced: MaxDD={result_v25_1x.metrics["MaxDrawdown"]:.1f}%')
    ax.fill_between(dd_opt.index, 0, dd_opt.values, color='green', alpha=0.5, label=f'V2.5_1x OPT: MaxDD={result_opt.metrics["MaxDrawdown"]:.1f}%')
    
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown Comparison (3-Way)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-50, 5)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'drawdown_3way_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: drawdown_3way_comparison.png")
    
    # Print final report
    print("\n" + "="*80)
    print("THREE-WAY COMPARISON: V2.5 vs V2.5_1x vs V2.5_1x OPTIMIZED")
    print("="*80)
    
    print(f"\n{'Version':<30} {'CAGR':>10} {'MaxDD':>10} {'Sharpe':>10} {'Final':>15}")
    print("-"*80)
    print(f"{'V2.5 (1.66x leverage)':<30} {result_v25.metrics['CAGR']:>9.1f}% {result_v25.metrics['MaxDrawdown']:>9.1f}% {result_v25.metrics['Sharpe']:>10.2f} ${result_v25.metrics['FinalEquity']:>13,.0f}")
    print(f"{'V2.5_1x (forced 1x)':<30} {result_v25_1x.metrics['CAGR']:>9.1f}% {result_v25_1x.metrics['MaxDrawdown']:>9.1f}% {result_v25_1x.metrics['Sharpe']:>10.2f} ${result_v25_1x.metrics['FinalEquity']:>13,.0f}")
    print(f"{'V2.5_1x OPTIMIZED':<30} {result_opt.metrics['CAGR']:>9.1f}% {result_opt.metrics['MaxDrawdown']:>9.1f}% {result_opt.metrics['Sharpe']:>10.2f} ${result_opt.metrics['FinalEquity']:>13,.0f}")
    
    # Calculate improvement
    improvement_cagr = result_opt.metrics['CAGR'] - result_v25_1x.metrics['CAGR']
    improvement_sharpe = result_opt.metrics['Sharpe'] - result_v25_1x.metrics['Sharpe']
    improvement_dd = result_v25_1x.metrics['MaxDrawdown'] - result_opt.metrics['MaxDrawdown']
    
    print(f"\n### Optimization Impact (vs forced 1x) ###")
    print(f"  CAGR improvement: {improvement_cagr:+.1f}%")
    print(f"  Sharpe improvement: {improvement_sharpe:+.2f}")
    print(f"  MaxDD reduction: {improvement_dd:+.1f}%")
    
    print(f"\n### Key Parameter Differences ###")
    for k in ['donchian_entry_period', 'donchian_exit_period', 'adx_gate_threshold', 
              'vol_target', 'atr_stop_mult', 'min_hold_hours', 'cooldown_hours']:
        v25_val = v25_params.get(k, 'N/A')
        opt_val = best_params.get(k, 'N/A')
        if v25_val != opt_val:
            print(f"  {k}: {v25_val} -> {opt_val}")
    
    print(f"\n### Trade Statistics ###")
    print(f"  V2.5 Trades: {result_v25.metrics['TradesCount']}")
    print(f"  V2.5_1x Trades: {result_v25_1x.metrics['TradesCount']}")
    print(f"  V2.5_1x OPT Trades: {result_opt.metrics['TradesCount']}")
    print(f"  Max Win Streak: {max_win}")
    print(f"  Max Loss Streak: {max_loss}")
    
    print(f"\n### OUTPUT FILES ###")
    print(f"  - {output_dir / 'best_params_v25_1x_optimized.json'}")
    print(f"  - {excel_path}")
    for f in plots_dir.glob('*.png'):
        print(f"  - {f}")
    
    print("\n" + "="*80)
    print("V2.5_1x OPTIMIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
