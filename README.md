# ETH Empirical Trading Strategy - V2/V2.5 Adaptive Walk-Forward Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust, production-ready cryptocurrency trading strategy backtesting framework with adaptive walk-forward optimization. Designed for ETH-USD trading with comprehensive risk management.

## 🎯 Strategy Overview

This project implements a **Donchian Channel breakout strategy** with multiple regime filters and volatility-targeted position sizing. Two versions are available:

| Version | Focus | CAGR | MaxDD | Sharpe |
|---------|-------|------|-------|--------|
| **V2** | Balanced Performance | 63.7% | 41.8% | 1.32 |
| **V2.5** | Stability & Tail Risk | 71.4% | 39.1% | 1.40 |

**Recommendation:** Use **V2.5** for production - it outperforms V2 in both return AND risk metrics.

## 📊 Key Features

### Strategy Components
- **Entry:** Donchian Channel breakout (130-hour period)
- **Exit:** Shorter Donchian exit + ATR trailing stop
- **Regime Filters:**
  - ADX gate (trend strength > 25)
  - Choppiness Index filter (< 70)
  - Regime MA filter (156-hour)
  - Crash volatility detection
- **Position Sizing:** Volatility-targeted (49% annualized vol target)
- **Leverage:** Capped at 1.66x

### Risk Management
- Maximum Drawdown constraint: 45%
- Minimum hold period: 81 hours
- Cooldown after exit: 18 hours
- Transaction costs: 4 bps commission + 5 bps slippage

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AlexBBno1/MetaGPT_ETH-Backtest-Agent.git
cd MetaGPT_ETH-Backtest-Agent

# Install dependencies
pip install -r requirements.txt
```

### Run Backtest with Best Parameters

```bash
# Run V2.5 (recommended)
python export_trade_details.py

# Run full optimization (takes ~10-15 minutes)
python run_v2_adaptive_complete.py --stage1-trials 30 --stage2-trials 10
```

### Run V2.5 Stability Optimization

```bash
python run_v2_5_stability.py
```

## 📁 Project Structure

```
eth_empirical_routes_5y/
├── src/
│   ├── backtester.py              # Core backtesting engine
│   ├── data_loader.py             # ETH data loader with caching
│   ├── optimizer_v2.py            # Base optimizer
│   ├── optimizer_v2_adaptive.py   # Adaptive walk-forward optimizer
│   └── strategies/
│       └── robust_strategy_v2.py  # Trading strategy implementation
├── outputs_v2/
│   ├── best_params_v2.json        # V2 optimized parameters
│   ├── trade_report_v2.xlsx       # Detailed trade report
│   └── plots/                     # Performance charts
├── outputs_v2_5/
│   ├── best_params_v25.json       # V2.5 optimized parameters
│   ├── V2_vs_V25_COMPARISON.md    # Detailed comparison
│   ├── v25_full_report.xlsx       # V2.5 trade report
│   └── plots/                     # Comparison charts
├── run_v2_adaptive_complete.py    # Main optimization script
├── run_v2_5_stability.py          # V2.5 stability optimization
├── export_trade_details.py        # Generate trade reports
└── README.md
```

## 📈 Performance Results

### Equity Growth (2020-01 to 2025-11)

| Metric | V2 | V2.5 |
|--------|-----|------|
| Initial Capital | $15,000 | $15,000 |
| Final Equity | $277,365 | $363,559 |
| Total Return | 1,749% | 2,324% |
| CAGR | 63.7% | 71.4% |
| Max Drawdown | 41.8% | 39.1% |
| Sharpe Ratio | 1.32 | 1.40 |
| Win Rate | 38.5% | 39.9% |
| Total Trades | 400 | 394 |

### Rolling 12-Month Stability

| Statistic | V2 | V2.5 |
|-----------|-----|------|
| Median CAGR | 39.2% | 49.1% |
| CAGR Std Dev | 68.6% | 61.4% |
| Worst MaxDD | 41.8% | 39.1% |

### Crash Period Performance

| Event | ETH | V2 | V2.5 |
|-------|-----|-----|------|
| COVID Crash (2020-03) | -39.4% | -12.0% | **+15.7%** |
| Luna/3AC (2022-06) | -44.1% | +39.0% | **+51.8%** |
| May 2021 Crash | -15.0% | +9.3% | +2.1% |

## ⚙️ Configuration

### Best Parameters (V2.5)

```json
{
  "donchian_entry_period": 130,
  "donchian_exit_period": 74,
  "adx_gate_threshold": 25,
  "chop_threshold": 70,
  "vol_target": 0.49,
  "regime_ma_period": 156,
  "min_hold_hours": 81,
  "leverage_cap": 1.66,
  "atr_stop_mult": 4.09,
  "crash_vol_mult": 2.55,
  "position_change_threshold": 0.41,
  "cooldown_hours": 18,
  "disable_regime_gate": false
}
```

### Backtest Configuration

```python
BacktestConfig(
    initial_capital=15000.0,
    commission_bps=4.0,      # 0.04% per trade
    slippage_bps=5.0,        # 0.05% slippage
    leverage=1.66,
    max_leverage=1.66
)
```

## 🔬 Optimization Process

### Stage 1: Adaptive Coarse Search
- Progressive constraint relaxation (3 levels)
- 30 trials per level
- Median pruner for early stopping

### Stage 2: Strict Refine Search
- Top 3 candidates from Stage 1
- Strict constraints (MaxDD ≤ 45%, Sharpe ≥ 0.7)
- 10 refinement trials per candidate

### V2.5 Stability Focus
- Only 5 parameters tuned (±15% range)
- Stability-oriented scoring function
- Rolling 12M performance tracking
- Crash period isolation testing

## 📊 Reports Generated

### Excel Reports
- **All Trades:** Entry/exit prices, size, leverage, P&L, streaks
- **Monthly P&L:** Monthly returns with running equity
- **Yearly Summary:** Annual performance breakdown
- **Streak Analysis:** Max win/loss streaks

### Charts
- Equity curve comparison
- Drawdown analysis
- Rolling performance
- Trade distribution
- Price with entry/exit markers

## 🛡️ Risk Disclaimers

- Past performance does not guarantee future results
- Cryptocurrency trading involves substantial risk
- This is research/educational software, not financial advice
- Backtest results may not reflect real trading conditions

## 📝 Changelog

### V2.5 (2025-12-30)
- Stability-oriented optimization
- Lower MaxDD (39.1% vs 41.8%)
- Higher Sharpe (1.40 vs 1.32)
- Better crash performance

### V2 (2025-12-29)
- Adaptive walk-forward optimization
- Robust regime filtering
- Transaction cost modeling

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

---

**Built with ❤️ for quantitative trading research**
