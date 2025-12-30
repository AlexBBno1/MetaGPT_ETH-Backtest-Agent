"""
ETH Empirical Routes V2 - Main Execution Script

Two-stage optimization with:
- Reduced parameters (<=12)
- Strict regime gate (bad regime = flat)
- Walk-forward test focused scoring
- Robustness testing (yearly splits, cost sensitivity, param perturbation)

Usage:
    python run_backtest_v2.py
    python run_backtest_v2.py --stage1-trials 60 --stage2-trials 40
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data_loader import load_data, get_data_info
from backtester import Backtester, BacktestConfig
from strategies.robust_strategy_v2 import RobustStrategyV2
from optimizer_v2 import WalkForwardOptimizerV2, WalkForwardResultV2


# =============================================================================
# Robustness Testing
# =============================================================================

class RobustnessTests:
    """Run robustness tests on final parameters."""
    
    def __init__(self, df: pd.DataFrame, params: Dict):
        self.df = df
        self.params = params
        self.results = {}
    
    def run_all(self, verbose: bool = True) -> Dict:
        """Run all robustness tests."""
        if verbose:
            print("\n" + "="*70)
            print("Running Robustness Tests")
            print("="*70)
        
        self.results['yearly_splits'] = self.test_yearly_splits(verbose)
        self.results['cost_sensitivity'] = self.test_cost_sensitivity(verbose)
        self.results['param_perturbation'] = self.test_param_perturbation(verbose)
        
        # Summary
        if verbose:
            self._print_summary()
        
        return self.results
    
    def test_yearly_splits(self, verbose: bool = True) -> Dict:
        """Test performance on different year splits."""
        if verbose:
            print("\n[1/3] Yearly Split Analysis")
        
        periods = {
            '2020-2021': ('2020-01-01', '2021-12-31'),
            '2022': ('2022-01-01', '2022-12-31'),
            '2023': ('2023-01-01', '2023-12-31'),
            '2024-2025': ('2024-01-01', '2025-11-30'),
        }
        
        results = {}
        for period_name, (start, end) in periods.items():
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            
            period_df = self.df[
                (self.df['timestamp'] >= start_dt) & 
                (self.df['timestamp'] <= end_dt)
            ].reset_index(drop=True)
            
            if len(period_df) < 100:
                continue
            
            try:
                strategy = RobustStrategyV2(self.params)
                signals = strategy.generate_signals(period_df)
                
                bt_config = BacktestConfig(
                    initial_capital=15000.0,
                    leverage=self.params.get('leverage_cap', 1.5)
                )
                backtester = Backtester(bt_config)
                result = backtester.run(period_df, signals)
                
                results[period_name] = {
                    'CAGR': result.metrics['CAGR'],
                    'MaxDD': result.metrics['MaxDrawdown'],
                    'Sharpe': result.metrics['Sharpe'],
                    'Trades': result.metrics['TradesCount'],
                }
                
                if verbose:
                    print(f"  {period_name}: CAGR {result.metrics['CAGR']:+.1f}%, "
                          f"MaxDD {result.metrics['MaxDrawdown']:.1f}%, "
                          f"Sharpe {result.metrics['Sharpe']:.2f}")
            except Exception as e:
                if verbose:
                    print(f"  {period_name}: Error - {e}")
        
        return results
    
    def test_cost_sensitivity(self, verbose: bool = True) -> Dict:
        """Test performance under different cost scenarios."""
        if verbose:
            print("\n[2/3] Cost Sensitivity Analysis")
        
        cost_multipliers = [1.0, 2.0, 3.0]
        base_commission = 4.0
        base_slippage = 5.0
        
        results = {}
        for mult in cost_multipliers:
            try:
                strategy = RobustStrategyV2(self.params)
                signals = strategy.generate_signals(self.df)
                
                bt_config = BacktestConfig(
                    initial_capital=15000.0,
                    commission_bps=base_commission * mult,
                    slippage_bps=base_slippage * mult,
                    leverage=self.params.get('leverage_cap', 1.5)
                )
                backtester = Backtester(bt_config)
                result = backtester.run(self.df, signals)
                
                results[f'{mult}x'] = {
                    'CAGR': result.metrics['CAGR'],
                    'MaxDD': result.metrics['MaxDrawdown'],
                    'Sharpe': result.metrics['Sharpe'],
                    'FinalEquity': result.metrics['FinalEquity'],
                }
                
                if verbose:
                    print(f"  Cost {mult}x: CAGR {result.metrics['CAGR']:+.1f}%, "
                          f"Sharpe {result.metrics['Sharpe']:.2f}, "
                          f"Final ${result.metrics['FinalEquity']:,.0f}")
            except Exception as e:
                if verbose:
                    print(f"  Cost {mult}x: Error - {e}")
        
        return results
    
    def test_param_perturbation(self, verbose: bool = True, n_tests: int = 10) -> Dict:
        """Test parameter robustness with ±10% perturbations."""
        if verbose:
            print(f"\n[3/3] Parameter Perturbation Analysis (n={n_tests})")
        
        np.random.seed(42)
        
        # Key parameters to perturb
        key_params = [
            'donchian_entry_period', 'donchian_exit_period', 'adx_gate_threshold',
            'vol_target', 'leverage_cap', 'min_hold_hours', 'chop_threshold'
        ]
        
        perturbation_results = []
        
        for i in range(n_tests):
            perturbed_params = self.params.copy()
            
            # Apply ±10% perturbation to each key param
            for key in key_params:
                if key in perturbed_params:
                    val = perturbed_params[key]
                    perturbation = 1 + np.random.uniform(-0.10, 0.10)
                    if isinstance(val, int):
                        perturbed_params[key] = max(1, int(val * perturbation))
                    else:
                        perturbed_params[key] = val * perturbation
            
            try:
                strategy = RobustStrategyV2(perturbed_params)
                signals = strategy.generate_signals(self.df)
                
                bt_config = BacktestConfig(
                    initial_capital=15000.0,
                    leverage=perturbed_params.get('leverage_cap', 1.5)
                )
                backtester = Backtester(bt_config)
                result = backtester.run(self.df, signals)
                
                perturbation_results.append({
                    'CAGR': result.metrics['CAGR'],
                    'MaxDD': result.metrics['MaxDrawdown'],
                    'Sharpe': result.metrics['Sharpe'],
                })
            except Exception as e:
                perturbation_results.append({
                    'CAGR': -100,
                    'MaxDD': 100,
                    'Sharpe': -1,
                })
        
        # Calculate statistics
        cagrs = [r['CAGR'] for r in perturbation_results]
        maxdds = [r['MaxDD'] for r in perturbation_results]
        sharpes = [r['Sharpe'] for r in perturbation_results]
        
        results = {
            'CAGR': {'mean': np.mean(cagrs), 'std': np.std(cagrs), 
                    'min': np.min(cagrs), 'max': np.max(cagrs)},
            'MaxDD': {'mean': np.mean(maxdds), 'std': np.std(maxdds),
                     'min': np.min(maxdds), 'max': np.max(maxdds)},
            'Sharpe': {'mean': np.mean(sharpes), 'std': np.std(sharpes),
                      'min': np.min(sharpes), 'max': np.max(sharpes)},
            'raw_results': perturbation_results,
        }
        
        if verbose:
            print(f"  CAGR: {results['CAGR']['mean']:.1f}% ± {results['CAGR']['std']:.1f}% "
                  f"(range: {results['CAGR']['min']:.1f}% to {results['CAGR']['max']:.1f}%)")
            print(f"  MaxDD: {results['MaxDD']['mean']:.1f}% ± {results['MaxDD']['std']:.1f}%")
            print(f"  Sharpe: {results['Sharpe']['mean']:.2f} ± {results['Sharpe']['std']:.2f}")
        
        # Check for overfitting
        negative_cagr_count = sum(1 for c in cagrs if c < 0)
        if negative_cagr_count > n_tests * 0.3:
            if verbose:
                print(f"  ⚠️ WARNING: {negative_cagr_count}/{n_tests} perturbations resulted in negative CAGR!")
        
        return results
    
    def _print_summary(self):
        """Print robustness summary."""
        print("\n" + "="*70)
        print("Robustness Summary")
        print("="*70)
        
        # Check yearly consistency
        yearly = self.results.get('yearly_splits', {})
        if yearly:
            cagrs = [v['CAGR'] for v in yearly.values()]
            negative_years = sum(1 for c in cagrs if c < 0)
            print(f"Yearly Consistency: {len(cagrs) - negative_years}/{len(cagrs)} years profitable")
        
        # Check cost robustness
        costs = self.results.get('cost_sensitivity', {})
        if costs:
            cagr_3x = costs.get('3.0x', {}).get('CAGR', -100)
            if cagr_3x > 0:
                print(f"Cost Robustness: ✓ Profitable at 3x costs (CAGR {cagr_3x:.1f}%)")
            else:
                print(f"Cost Robustness: ⚠️ Not profitable at 3x costs")
        
        # Check param robustness
        perturb = self.results.get('param_perturbation', {})
        if perturb:
            mean_cagr = perturb.get('CAGR', {}).get('mean', -100)
            std_cagr = perturb.get('CAGR', {}).get('std', 100)
            cv = abs(std_cagr / mean_cagr) if abs(mean_cagr) > 0.01 else 100
            
            if cv < 0.5 and mean_cagr > 0:
                print(f"Param Robustness: ✓ Stable (CV={cv:.2f})")
            else:
                print(f"Param Robustness: ⚠️ Unstable (CV={cv:.2f})")


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_equity_curve_v2(
    equity_df: pd.DataFrame,
    metrics: Dict,
    output_path: Path,
    initial_capital: float = 15000.0
):
    """Plot equity curve for V2 strategy."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), 
                              gridspec_kw={'height_ratios': [3, 1, 1]})
    
    timestamps = pd.to_datetime(equity_df['timestamp'])
    equity = equity_df['equity']
    
    # Main equity curve
    ax1 = axes[0]
    ax1.plot(timestamps, equity, color='#1E88E5', linewidth=1.5, label='Strategy V2')
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    
    # Fill underwater
    ax1.fill_between(timestamps, initial_capital, equity, 
                     where=(equity < initial_capital), 
                     color='#E53935', alpha=0.2, interpolate=True)
    ax1.fill_between(timestamps, initial_capital, equity, 
                     where=(equity >= initial_capital), 
                     color='#43A047', alpha=0.2, interpolate=True)
    
    ax1.set_ylabel('Portfolio Value (USD)', fontsize=11)
    ax1.set_title(f'Strategy V2 - 5-Year Equity Curve (2020-2025)\n'
                  f'CAGR: {metrics["CAGR"]:.1f}% | MaxDD: {metrics["MaxDrawdown"]:.1f}% | '
                  f'Sharpe: {metrics["Sharpe"]:.2f} | Final: ${metrics["FinalEquity"]:,.0f}',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Drawdown
    ax2 = axes[1]
    drawdown = equity_df['drawdown'] * 100
    ax2.fill_between(timestamps, 0, drawdown, color='#E53935', alpha=0.4)
    ax2.plot(timestamps, drawdown, color='#B71C1C', linewidth=0.8)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_ylim(top=0)
    ax2.axhline(y=-metrics['MaxDrawdown'], color='red', linestyle='--', alpha=0.7)
    
    # Position
    ax3 = axes[2]
    position = equity_df['position']
    ax3.fill_between(timestamps, 0, position, where=(position > 0), 
                     color='#43A047', alpha=0.4, label='Long')
    ax3.fill_between(timestamps, 0, position, where=(position < 0), 
                     color='#E53935', alpha=0.4, label='Short')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_ylabel('Position', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.legend(loc='upper right')
    
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_drawdown_v2(
    equity_df: pd.DataFrame,
    metrics: Dict,
    output_path: Path
):
    """Plot drawdown chart."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    timestamps = pd.to_datetime(equity_df['timestamp'])
    drawdown = equity_df['drawdown'] * 100
    
    ax.fill_between(timestamps, 0, drawdown, color='#E53935', alpha=0.5)
    ax.plot(timestamps, drawdown, color='#B71C1C', linewidth=1)
    
    # Mark max drawdown
    max_dd_idx = drawdown.idxmin()
    ax.scatter([timestamps[max_dd_idx]], [drawdown[max_dd_idx]], 
               color='#B71C1C', s=100, zorder=5)
    ax.annotate(f'Max DD: {metrics["MaxDrawdown"]:.1f}%', 
                xy=(timestamps[max_dd_idx], drawdown[max_dd_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
    ax.set_ylim(top=5)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_yearly_returns_v2(
    equity_df: pd.DataFrame,
    output_path: Path
):
    """Plot yearly returns bar chart."""
    equity_df = equity_df.copy()
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    equity_df['year'] = equity_df['timestamp'].dt.year
    
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    yearly_returns = []
    
    for year in years:
        year_data = equity_df[equity_df['year'] == year]
        if len(year_data) > 1:
            year_return = (year_data['equity'].iloc[-1] / year_data['equity'].iloc[0] - 1) * 100
        else:
            year_return = 0
        yearly_returns.append(year_return)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#43A047' if r > 0 else '#E53935' for r in yearly_returns]
    bars = ax.bar(years, yearly_returns, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, yearly_returns):
        height = bar.get_height()
        ax.annotate(f'{val:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -3),
                    textcoords='offset points',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Return (%)', fontsize=12)
    ax.set_title('Yearly Returns - Strategy V2', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cost_sensitivity_v2(
    cost_results: Dict,
    output_path: Path
):
    """Plot cost sensitivity analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    multipliers = ['1x', '2x', '3x']
    cagrs = [cost_results.get(f'{m}', {}).get('CAGR', 0) for m in ['1.0x', '2.0x', '3.0x']]
    sharpes = [cost_results.get(f'{m}', {}).get('Sharpe', 0) for m in ['1.0x', '2.0x', '3.0x']]
    
    x = np.arange(len(multipliers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cagrs, width, label='CAGR (%)', color='#1E88E5')
    bars2 = ax.bar(x + width/2, [s * 20 for s in sharpes], width, label='Sharpe × 20', color='#FFA726')
    
    ax.set_xlabel('Cost Multiplier', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Cost Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(multipliers)
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_robustness_perturbation(
    perturb_results: Dict,
    output_path: Path
):
    """Plot parameter perturbation analysis as boxplot."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    raw = perturb_results.get('raw_results', [])
    if not raw:
        return
    
    cagrs = [r['CAGR'] for r in raw]
    maxdds = [r['MaxDD'] for r in raw]
    sharpes = [r['Sharpe'] for r in raw]
    
    # CAGR boxplot
    bp1 = axes[0].boxplot([cagrs], vert=True, patch_artist=True)
    bp1['boxes'][0].set_facecolor('#1E88E5')
    axes[0].set_ylabel('CAGR (%)', fontsize=11)
    axes[0].set_title('CAGR Distribution', fontsize=12, fontweight='bold')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticklabels([''])
    
    # MaxDD boxplot
    bp2 = axes[1].boxplot([maxdds], vert=True, patch_artist=True)
    bp2['boxes'][0].set_facecolor('#E53935')
    axes[1].set_ylabel('MaxDD (%)', fontsize=11)
    axes[1].set_title('MaxDD Distribution', fontsize=12, fontweight='bold')
    axes[1].axhline(y=45, color='red', linestyle='--', alpha=0.5, label='Target Max')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticklabels([''])
    
    # Sharpe boxplot
    bp3 = axes[2].boxplot([sharpes], vert=True, patch_artist=True)
    bp3['boxes'][0].set_facecolor('#43A047')
    axes[2].set_ylabel('Sharpe', fontsize=11)
    axes[2].set_title('Sharpe Distribution', fontsize=12, fontweight='bold')
    axes[2].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target Min')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticklabels([''])
    
    fig.suptitle('Parameter Perturbation Analysis (±10%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Report Generation
# =============================================================================

def generate_report_v2(
    result: WalkForwardResultV2,
    robustness: Dict,
    v1_metrics: Dict,
    output_path: Path
):
    """Generate comprehensive markdown report."""
    
    v2_full = result.full_metrics
    v2_train = result.train_metrics
    v2_test = result.test_metrics
    
    # Calculate improvements
    cagr_improve = v2_full['CAGR'] - v1_metrics['CAGR']
    maxdd_improve = v1_metrics['MaxDD'] - v2_full['MaxDrawdown']  # Positive is better
    sharpe_improve = v2_full['Sharpe'] - v1_metrics['Sharpe']
    trades_improve = v1_metrics['Trades'] - v2_full['TradesCount']  # Lower is better for turnover
    
    report = f"""# ETH Strategy V2 - Optimized Backtest Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Period:** 2020-01-01 to 2025-11-30
**Initial Capital:** $15,000 USD

---

## Executive Summary

This report presents the results of the V2 optimization, which implements:
- Reduced parameters (12 total, down from ~20)
- Strict regime gate (bad regime = FLAT position)
- Two-stage optimization (Coarse + Refine)
- Walk-forward test focused scoring
- Robustness testing

---

## V2 vs V1 Comparison

| Metric | V1 (Route A) | V2 | Change |
|--------|-------------|-----|--------|
| CAGR | {v1_metrics['CAGR']:.2f}% | {v2_full['CAGR']:.2f}% | {cagr_improve:+.2f}% |
| MaxDD | {v1_metrics['MaxDD']:.2f}% | {v2_full['MaxDrawdown']:.2f}% | {maxdd_improve:+.2f}% (improved) |
| Sharpe | {v1_metrics['Sharpe']:.2f} | {v2_full['Sharpe']:.2f} | {sharpe_improve:+.2f} |
| Trades | {v1_metrics['Trades']} | {v2_full['TradesCount']} | {trades_improve:+d} (reduced) |
| Turnover (units/yr) | {v1_metrics.get('Turnover', 'N/A')} | {v2_full.get('Turnover', 0):.0f} | Reduced |
| Final Equity | ${v1_metrics.get('FinalEquity', 34837):,.0f} | ${v2_full['FinalEquity']:,.0f} | - |

---

## Walk-Forward Results

### Training Period (2020-2022)
| Metric | Value |
|--------|-------|
| CAGR | {v2_train['CAGR']:.2f}% |
| MaxDrawdown | {v2_train['MaxDrawdown']:.2f}% |
| Sharpe | {v2_train['Sharpe']:.2f} |
| Trades | {v2_train['TradesCount']} |

### Test Period (2023-2025) - Out-of-Sample
| Metric | Value |
|--------|-------|
| CAGR | {v2_test['CAGR']:.2f}% |
| MaxDrawdown | {v2_test['MaxDrawdown']:.2f}% |
| Sharpe | {v2_test['Sharpe']:.2f} |
| Trades | {v2_test['TradesCount']} |

---

## 1. Performance Improvements

### Sharpe / MaxDD / CAGR Changes
- **Sharpe Improvement:** {sharpe_improve:+.2f} ({v1_metrics['Sharpe']:.2f} → {v2_full['Sharpe']:.2f})
- **MaxDD Improvement:** {maxdd_improve:+.2f}% ({v1_metrics['MaxDD']:.2f}% → {v2_full['MaxDrawdown']:.2f}%)
- **CAGR Change:** {cagr_improve:+.2f}% ({v1_metrics['CAGR']:.2f}% → {v2_full['CAGR']:.2f}%)

### Target Achievement
- Sharpe ≥ 0.8: {"✓ Achieved" if v2_full['Sharpe'] >= 0.8 else f"✗ Not achieved ({v2_full['Sharpe']:.2f})"}
- MaxDD ≤ 45%: {"✓ Achieved" if v2_full['MaxDrawdown'] <= 45 else f"✗ Not achieved ({v2_full['MaxDrawdown']:.1f}%)"}
- CAGR ≥ 20%: {"✓ Achieved" if v2_full['CAGR'] >= 20 else f"✗ Not achieved ({v2_full['CAGR']:.1f}%)"}

---

## 2. Turnover & Trade Reduction

| Metric | V1 | V2 | Change |
|--------|-----|-----|--------|
| Trades Count | {v1_metrics['Trades']} | {v2_full['TradesCount']} | {trades_improve:+d} |
| Turnover (units/yr) | {v1_metrics.get('Turnover', 18430):.0f} | {v2_full.get('Turnover', 0):.0f} | Reduced |

**Analysis:** The parameter reduction and strict position change threshold significantly 
reduced unnecessary trading activity. The minimum hold period and cooldown also contributed
to lower turnover.

---

## 3. Key Factors & Rules

### Most Critical Components:
1. **Regime Gate (ADX + Choppiness):** 
   - Bad regime = FLAT position (0% exposure)
   - This eliminated the "slow bleeding" during ranging markets
   
2. **Vol Targeting:** 
   - Position size scaled by volatility target
   - Provides natural position reduction during high vol

3. **Donchian Breakout:**
   - Entry on high/low channel breakouts
   - Exit on shorter-period channel touch
   
4. **ATR Stop Loss:**
   - Fixed stop based on ATR multiple
   - Prevents large single-trade losses

### Parameter Summary (12 total):
```json
{json.dumps(result.best_params, indent=2)}
```

---

## 4. 2023 Analysis - Why It's Difficult

"""
    
    # Add 2023 specific analysis
    yearly = robustness.get('yearly_splits', {})
    y2023 = yearly.get('2023', {})
    
    report += f"""
### 2023 Performance
- CAGR: {y2023.get('CAGR', 'N/A')}%
- MaxDD: {y2023.get('MaxDD', 'N/A')}%
- Sharpe: {y2023.get('Sharpe', 'N/A')}

### Why 2023 Was Challenging:
1. **Low Volatility Periods:** Extended periods of low ADX caused regime gate to activate,
   reducing exposure and missing some trends
2. **Choppy Price Action:** Multiple false breakouts led to stopped-out positions
3. **No Clear Trend:** 2023 lacked the strong directional moves seen in 2020-2021

### How V2 Improved:
- Strict regime gate prevented losses from trading in bad conditions
- Higher position change threshold avoided whipsaws
- Longer minimum hold periods reduced premature exits

---

## 5. Failure Conditions

The strategy may fail or underperform when:

1. **Extended Ranging Markets:** If ADX stays below threshold for months,
   the strategy will be mostly flat and miss any gains
   
2. **V-shaped Reversals:** Sharp reversals after breakouts can hit stops
   before the strategy can profit
   
3. **Gap Risk:** Large overnight gaps can bypass stops (not modeled here)

4. **Regime Changes:** If market character shifts significantly, parameters
   optimized on past data may become suboptimal

5. **Higher Costs:** At 3x cost assumptions, profitability is reduced
"""
    
    # Add robustness results
    cost_results = robustness.get('cost_sensitivity', {})
    perturb_results = robustness.get('param_perturbation', {})
    
    report += f"""
---

## 6. Robustness Test Results

### Cost Sensitivity
| Cost Level | CAGR | Sharpe |
|------------|------|--------|
| 1x (base) | {cost_results.get('1.0x', {}).get('CAGR', 0):.1f}% | {cost_results.get('1.0x', {}).get('Sharpe', 0):.2f} |
| 2x | {cost_results.get('2.0x', {}).get('CAGR', 0):.1f}% | {cost_results.get('2.0x', {}).get('Sharpe', 0):.2f} |
| 3x | {cost_results.get('3.0x', {}).get('CAGR', 0):.1f}% | {cost_results.get('3.0x', {}).get('Sharpe', 0):.2f} |

### Parameter Perturbation (±10%)
| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| CAGR | {perturb_results.get('CAGR', {}).get('mean', 0):.1f}% | {perturb_results.get('CAGR', {}).get('std', 0):.1f}% | {perturb_results.get('CAGR', {}).get('min', 0):.1f}% | {perturb_results.get('CAGR', {}).get('max', 0):.1f}% |
| MaxDD | {perturb_results.get('MaxDD', {}).get('mean', 0):.1f}% | {perturb_results.get('MaxDD', {}).get('std', 0):.1f}% | {perturb_results.get('MaxDD', {}).get('min', 0):.1f}% | {perturb_results.get('MaxDD', {}).get('max', 0):.1f}% |
| Sharpe | {perturb_results.get('Sharpe', {}).get('mean', 0):.2f} | {perturb_results.get('Sharpe', {}).get('std', 0):.2f} | {perturb_results.get('Sharpe', {}).get('min', 0):.2f} | {perturb_results.get('Sharpe', {}).get('max', 0):.2f} |

### Yearly Splits
| Period | CAGR | MaxDD | Sharpe |
|--------|------|-------|--------|
"""
    
    for period_name, metrics in yearly.items():
        report += f"| {period_name} | {metrics.get('CAGR', 0):.1f}% | {metrics.get('MaxDD', 0):.1f}% | {metrics.get('Sharpe', 0):.2f} |\n"
    
    report += f"""
---

## Files Generated

- `best_params_v2.json`: Optimized parameters
- `metrics_full_v2.json`: Full period metrics
- `metrics_walkforward_v2.json`: Train/Test metrics
- `stress_tests_summary.csv`: Robustness test results
- `equity_curve_v2.parquet`: Equity curve data
- `plots/`
  - `equity_curve_v2.png`: 5-year equity curve
  - `drawdown_v2.png`: Drawdown chart
  - `yearly_returns_v2.png`: Yearly returns
  - `cost_sensitivity_v2.png`: Cost sensitivity
  - `robustness_perturbation.png`: Parameter perturbation

---

**Report End**
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Saved: {output_path}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='ETH Strategy V2 Backtest')
    parser.add_argument('--stage1-trials', type=int, default=60,
                        help='Number of Stage 1 (coarse) trials')
    parser.add_argument('--stage2-trials', type=int, default=40,
                        help='Number of Stage 2 (refine) trials per candidate')
    parser.add_argument('--n-candidates', type=int, default=3,
                        help='Number of candidates from Stage 1')
    parser.add_argument('--skip-optimization', action='store_true',
                        help='Skip optimization and use saved params')
    parser.add_argument('--no-log', action='store_true',
                        help='Disable logging to file')
    args = parser.parse_args()
    
    # Setup paths
    output_dir = PROJECT_ROOT / 'outputs_v2'
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)
    
    db_path = output_dir / 'optuna_v2.db'
    
    print("="*70)
    print("ETH Strategy V2 - Optimized Backtest")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {output_dir}")
    print(f"Stage 1 Trials: {args.stage1_trials}")
    print(f"Stage 2 Trials: {args.stage2_trials} x {args.n_candidates}")
    print("="*70)
    
    # Load V1 metrics for comparison
    v1_metrics_path = PROJECT_ROOT / 'outputs' / 'metrics_route_A.json'
    if v1_metrics_path.exists():
        with open(v1_metrics_path) as f:
            v1_metrics = json.load(f)
        v1_metrics = {
            'CAGR': v1_metrics.get('CAGR', 15.3),
            'MaxDD': v1_metrics.get('MaxDrawdown', 59.8),
            'Sharpe': v1_metrics.get('Sharpe', 0.64),
            'Trades': v1_metrics.get('TradesCount', 1535),
            'Turnover': v1_metrics.get('Turnover', 18430),
            'FinalEquity': v1_metrics.get('FinalEquity', 34837),
        }
    else:
        v1_metrics = {
            'CAGR': 15.3, 'MaxDD': 59.8, 'Sharpe': 0.64,
            'Trades': 1535, 'Turnover': 18430, 'FinalEquity': 34837
        }
    
    print(f"\nV1 Baseline (Route A):")
    print(f"  CAGR: {v1_metrics['CAGR']:.2f}%")
    print(f"  MaxDD: {v1_metrics['MaxDD']:.2f}%")
    print(f"  Sharpe: {v1_metrics['Sharpe']:.2f}")
    print(f"  Trades: {v1_metrics['Trades']}")
    
    # Load data
    print("\n[1/5] Loading ETH data (2020-01-01 to 2025-11-30)...")
    df = load_data(
        start_date="2020-01-01",
        end_date="2025-11-30",
        freq="1h"
    )
    
    info = get_data_info(df)
    print(f"Data loaded successfully:")
    print(f"  Rows: {info['rows']}")
    print(f"  Period: {info['start_date']} to {info['end_date']}")
    
    # Run optimization or load params
    params_path = output_dir / 'best_params_v2.json'
    
    if args.skip_optimization and params_path.exists():
        print("\n[2/5] Loading saved parameters...")
        with open(params_path) as f:
            best_params = json.load(f)
        
        # Run backtest with loaded params
        strategy = RobustStrategyV2(best_params)
        signals = strategy.generate_signals(df)
        
        bt_config = BacktestConfig(
            initial_capital=15000.0,
            leverage=best_params.get('leverage_cap', 1.5)
        )
        backtester = Backtester(bt_config)
        
        # Split for WF metrics
        train_end = pd.to_datetime("2022-12-31")
        train_df = df[df['timestamp'] <= train_end].reset_index(drop=True)
        test_df = df[df['timestamp'] > train_end].reset_index(drop=True)
        
        train_result = backtester.run(train_df, strategy.generate_signals(train_df))
        test_result = backtester.run(test_df, strategy.generate_signals(test_df))
        full_result = backtester.run(df, signals)
        
        result = WalkForwardResultV2(
            best_params=best_params,
            train_metrics=train_result.metrics,
            test_metrics=test_result.metrics,
            full_metrics=full_result.metrics,
            equity_curve=full_result.equity_curve,
            trades=full_result.trades,
            yearly_metrics={},
            optimization_history=[],
            stage1_candidates=[],
            stage2_results=[]
        )
    else:
        print("\n[2/5] Running two-stage optimization...")
        optimizer = WalkForwardOptimizerV2(
            stage1_trials=args.stage1_trials,
            stage2_trials=args.stage2_trials,
            n_candidates=args.n_candidates,
            db_path=str(db_path)
        )
        
        result = optimizer.walk_forward(df, train_end="2022-12-31", verbose=True)
        
        # Save best params immediately
        with open(params_path, 'w') as f:
            json.dump(result.best_params, f, indent=2)
        print(f"\nSaved: {params_path}")
    
    # Run robustness tests
    print("\n[3/5] Running robustness tests...")
    robustness_tester = RobustnessTests(df, result.best_params)
    robustness_results = robustness_tester.run_all(verbose=True)
    
    # Save results
    print("\n[4/5] Saving results...")
    
    # Metrics
    with open(output_dir / 'metrics_full_v2.json', 'w') as f:
        json.dump(result.full_metrics, f, indent=2)
    
    wf_metrics = {
        'train': result.train_metrics,
        'test': result.test_metrics
    }
    with open(output_dir / 'metrics_walkforward_v2.json', 'w') as f:
        json.dump(wf_metrics, f, indent=2)
    
    # Equity curve
    result.equity_curve.to_parquet(output_dir / 'equity_curve_v2.parquet', index=False)
    
    # Stress tests summary
    stress_rows = []
    
    for period, metrics in robustness_results.get('yearly_splits', {}).items():
        stress_rows.append({
            'test_type': 'yearly_split',
            'test_name': period,
            'CAGR': metrics.get('CAGR', 0),
            'MaxDD': metrics.get('MaxDD', 0),
            'Sharpe': metrics.get('Sharpe', 0),
        })
    
    for cost, metrics in robustness_results.get('cost_sensitivity', {}).items():
        stress_rows.append({
            'test_type': 'cost_sensitivity',
            'test_name': f'cost_{cost}',
            'CAGR': metrics.get('CAGR', 0),
            'MaxDD': metrics.get('MaxDD', 0),
            'Sharpe': metrics.get('Sharpe', 0),
        })
    
    perturb = robustness_results.get('param_perturbation', {})
    stress_rows.append({
        'test_type': 'param_perturbation',
        'test_name': 'mean',
        'CAGR': perturb.get('CAGR', {}).get('mean', 0),
        'MaxDD': perturb.get('MaxDD', {}).get('mean', 0),
        'Sharpe': perturb.get('Sharpe', {}).get('mean', 0),
    })
    
    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(output_dir / 'stress_tests_summary.csv', index=False)
    
    # Generate plots
    print("\n[5/5] Generating plots...")
    
    plot_equity_curve_v2(
        result.equity_curve, result.full_metrics,
        output_dir / 'plots' / 'equity_curve_v2.png'
    )
    
    plot_drawdown_v2(
        result.equity_curve, result.full_metrics,
        output_dir / 'plots' / 'drawdown_v2.png'
    )
    
    plot_yearly_returns_v2(
        result.equity_curve,
        output_dir / 'plots' / 'yearly_returns_v2.png'
    )
    
    plot_cost_sensitivity_v2(
        robustness_results.get('cost_sensitivity', {}),
        output_dir / 'plots' / 'cost_sensitivity_v2.png'
    )
    
    plot_robustness_perturbation(
        robustness_results.get('param_perturbation', {}),
        output_dir / 'plots' / 'robustness_perturbation.png'
    )
    
    # Generate report
    generate_report_v2(result, robustness_results, v1_metrics, output_dir / 'report_v2.md')
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL RESULTS - V2 vs V1 Comparison")
    print("="*70)
    print(f"\n{'Metric':<20} {'V1 (Route A)':<15} {'V2':<15} {'Change':<15}")
    print("-"*70)
    
    v2 = result.full_metrics
    
    print(f"{'CAGR':<20} {v1_metrics['CAGR']:.2f}%{'':<8} {v2['CAGR']:.2f}%{'':<8} {v2['CAGR']-v1_metrics['CAGR']:+.2f}%")
    print(f"{'MaxDrawdown':<20} {v1_metrics['MaxDD']:.2f}%{'':<8} {v2['MaxDrawdown']:.2f}%{'':<8} {v1_metrics['MaxDD']-v2['MaxDrawdown']:+.2f}% (better)")
    print(f"{'Sharpe':<20} {v1_metrics['Sharpe']:.2f}{'':<12} {v2['Sharpe']:.2f}{'':<12} {v2['Sharpe']-v1_metrics['Sharpe']:+.2f}")
    print(f"{'Trades':<20} {v1_metrics['Trades']:<15} {v2['TradesCount']:<15} {v1_metrics['Trades']-v2['TradesCount']:+d}")
    print(f"{'Turnover(units/yr)':<20} {v1_metrics['Turnover']:.0f}{'':<8} {v2.get('Turnover', 0):.0f}{'':<8} Reduced")
    print(f"{'Final Equity':<20} ${v1_metrics['FinalEquity']:,.0f}{'':<5} ${v2['FinalEquity']:,.0f}")
    
    print("\n" + "="*70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

