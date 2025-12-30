# ETH Strategy V2 - Adaptive Walk-Forward Optimization Report

**Generated:** 2025-12-30 11:36:32
**Period:** 2020-01-01 to 2025-11-30
**Initial Capital:** $15,000 USD

---

## Executive Summary

This report presents the results of the V2 **Adaptive** Walk-Forward Optimization (OPTIMIZED version).

### Final Status: PASSED Stage2 Strict Limits

---

## Relaxation Log

| Level | Name | min_trades/year | Valid/Total | Status |
|-------|------|-----------------|-------------|--------|
| 0 | standard | 30 | 9/10 | OK |

---

## Final Best Parameters

```json
{
  "donchian_entry_period": 130,
  "donchian_exit_period": 83,
  "adx_gate_threshold": 25,
  "vol_target": 0.4903572446872956,
  "leverage_cap": 1.6558603923728379,
  "min_hold_hours": 81,
  "cooldown_hours": 16,
  "position_change_threshold": 0.3919562505114017,
  "atr_stop_mult": 3.7771463511635632,
  "crash_vol_mult": 2.4696660642389117,
  "chop_threshold": 70,
  "regime_ma_period": 156,
  "disable_regime_gate": false
}
```

---

## Walk-Forward Results

### Training Period (2020-2022)
| Metric | Value |
|--------|-------|
| CAGR | 39.87% |
| MaxDrawdown | 36.54% |
| Sharpe | 0.95 |
| Trades | 207 |

### Test Period (2023-2025) - Out-of-Sample
| Metric | Value |
|--------|-------|
| CAGR | 74.55% |
| MaxDrawdown | 41.26% |
| Sharpe | 1.50 |
| Trades | 196 |

### Full Period (2020-2025)
| Metric | Value |
|--------|-------|
| CAGR | 56.89% |
| MaxDrawdown | 41.26% |
| Sharpe | 1.22 |
| Trades | 404 |
| Trades/Year | 68.3 |
| Turnover | 183 |
| Final Equity | $215,555 |

---

## V2 vs V1 Comparison

| Metric | V1 (Route A) | V2 | Change |
|--------|-------------|-----|--------|
| CAGR | 15.30% | 56.89% | +41.59% |
| MaxDD | 59.80% | 41.26% | +18.54% (improved) |
| Sharpe | 0.64 | 1.22 | +0.58 |
| Trades | 1535 | 404 | +1131 |

---

**Report End**
