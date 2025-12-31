# ETH Adaptive Walk-Forward Trading Strategy

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust cryptocurrency trading strategy with adaptive walk-forward optimization for ETH-USD.

---

## Strategy Versions

| Version | Leverage | CAGR | MaxDD | Sharpe | Final Equity |
|---------|----------|------|-------|--------|--------------|
| **V2** | 1.66x | 63.7% | 41.8% | 1.32 | $277,365 |
| **V2.5** | 1.66x | 71.4% | 39.1% | 1.40 | $363,559 |
| **V2.5_1x Optimized** | 1.0x | 53.8% | 27.1% | 1.43 | $191,554 |

> **Note:** V2.5_1x Optimized is specifically tuned for no-leverage trading.

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/AlexBBno1/MetaGPT_ETH-Backtest-Agent.git
cd MetaGPT_ETH-Backtest-Agent

# Install dependencies
pip install -r requirements.txt

# Run V2 optimization
python run_v2_adaptive_complete.py --stage1-trials 30 --stage2-trials 10

# Run V2.5 stability optimization
python run_v2_5_stability.py

# Run V2.5_1x optimized (no leverage)
python run_v2_5_1x_optimized.py

# Export trade details
python export_trade_details.py
python export_v25_1x_optimized_details.py
```

---

## Project Structure

```
eth_empirical_routes_5y/
├── src/                          # Core library
│   ├── backtester.py             # Backtesting engine
│   ├── data_loader.py            # Data loading with cache
│   ├── optimizer_v2.py           # Base optimizer
│   ├── optimizer_v2_adaptive.py  # Adaptive walk-forward
│   └── strategies/
│       └── robust_strategy_v2.py # Trading strategy
│
├── outputs_v2/                   # V2 Results
│   ├── best_params_v2.json       # Optimized parameters
│   ├── trade_details/            # Trade reports & charts
│   └── plots/                    # Performance charts
│
├── outputs_v2_5/                 # V2.5 Stability Results
│   ├── best_params_v25.json
│   ├── V2_vs_V25_COMPARISON.md   # Detailed comparison
│   └── plots/
│
├── outputs_v2_5_1x_optimized/    # V2.5 1x (No Leverage)
│   ├── best_params_v25_1x_optimized.json
│   ├── trade_report_v25_1x_optimized.xlsx
│   └── plots/
│
├── run_v2_adaptive_complete.py   # Main V2 optimizer
├── run_v2_5_stability.py         # V2.5 stability optimizer
├── run_v2_5_1x_optimized.py      # V2.5 1x optimizer
├── export_trade_details.py       # V2 trade export
├── export_v25_1x_optimized_details.py  # V2.5 1x export
├── README.md
└── requirements.txt
```

---

## Strategy Features

- **Entry:** Donchian Channel breakout
- **Exit:** Shorter Donchian + ATR trailing stop
- **Filters:** ADX, Choppiness Index, Regime MA
- **Risk:** Crash volatility detection, position sizing
- **Costs:** Commission 4 bps + Slippage 5 bps

---

## Performance Comparison

### V2.5 vs V2.5_1x Optimized

| Metric | V2.5 (1.66x) | V2.5_1x (1.0x) |
|--------|--------------|----------------|
| CAGR | 71.4% | 53.8% |
| MaxDrawdown | 39.1% | **27.1%** |
| Sharpe | 1.40 | **1.43** |
| Trades | 394 | 508 |
| Final Equity | $363,559 | $191,554 |

> V2.5_1x has **lower risk** (27% MaxDD) with **better risk-adjusted returns** (Sharpe 1.43)

---

## Key Parameters

### V2.5 (With Leverage)
```json
{
  "donchian_entry_period": 130,
  "donchian_exit_period": 74,
  "leverage_cap": 1.66,
  "vol_target": 0.49,
  "atr_stop_mult": 4.09
}
```

### V2.5_1x Optimized (No Leverage)
```json
{
  "donchian_entry_period": 91,
  "donchian_exit_period": 112,
  "leverage_cap": 1.0,
  "vol_target": 0.68,
  "atr_stop_mult": 3.01
}
```

---

## Output Reports

Each version generates:
- **Excel Report:** All trades, monthly/yearly PnL, streaks
- **Equity Curve:** Portfolio growth over time
- **Drawdown Chart:** Risk visualization
- **Trade Analysis:** PnL distribution, win rate by year

---

## Risk Disclaimers

- Past performance does not guarantee future results
- Cryptocurrency trading involves substantial risk
- This is research software, not financial advice

---

## License

MIT License
