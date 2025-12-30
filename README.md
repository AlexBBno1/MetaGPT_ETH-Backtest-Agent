# ETH Empirical Routes 5Y Backtest

基於 Factor Exploration 結果的 ETH 五年實證回測專案 (2020-01-01 ~ 2025-11-30)

## 專案概述

本專案實現三條策略路線的完整回測：

| 路線 | 類型 | 目標 CAGR | 槓桿 | MaxDD 上限 | Sharpe 目標 |
|------|------|-----------|------|-----------|-------------|
| A | 穩健型 | 25-30% | 1.0-1.2 | ≤30% | ≥0.9 |
| B | 折衷型 | 35-40% | 1.3-1.5 | ≤40% | ≥0.8 |
| C | 積極型 | 40-45% | 2.0-2.5 | ≤55% | - |

## 使用因子

基於 Factor Exploration 確認有效的因子組合：

1. **Trend Persistence** (ADX + Hurst) - 過濾盤整期假信號
2. **Chop Detector** - 避開高波動盤整
3. **Rules-Based Regime Sizing** - 動態倉位調整
4. **Volatility Breakout** - 捕捉突破機會
5. **Crash Protection** - 保護機制降低回撤

## 快速開始

### 方法一：PowerShell 腳本

```powershell
cd C:\Users\alhung\MetaGPT\projects\eth_empirical_routes_5y
.\run_all.ps1
```

### 方法二：Python 命令

```bash
# 安裝依賴
pip install -r requirements.txt

# 運行完整回測
python run_backtest.py

# 只運行特定路線
python run_backtest.py --route A

# 減少優化次數（快速測試）
python run_backtest.py --n-trials 20
```

## 輸出檔案

所有結果保存在 `outputs/` 目錄：

```
outputs/
├── metrics_route_A.json       # 路線 A 指標
├── metrics_route_B.json       # 路線 B 指標
├── metrics_route_C.json       # 路線 C 指標
├── equity_curve_route_A.parquet
├── equity_curve_route_B.parquet
├── equity_curve_route_C.parquet
├── plot_route_A.png           # 路線 A 曲線圖
├── plot_route_B.png
├── plot_route_C.png
├── routes_comparison.png      # 三條路線疊圖
├── yearly_returns.png         # 年度回報對比
└── empirical_routes_report.md # 完整報告
```

## 回測設定

- **標的**: ETH-USD
- **資料期間**: 2020-01-01 ~ 2025-11-30
- **頻率**: 1H (小時)
- **初始資金**: $15,000 USD
- **交易成本**: 
  - Commission: 4 bps
  - Slippage: 5 bps
- **Walk-Forward**: 
  - 訓練期: 2020-01-01 ~ 2022-12-31 (3年)
  - 測試期: 2023-01-01 ~ 2025-11-30 (近3年)

## 目錄結構

```
eth_empirical_routes_5y/
├── run_backtest.py            # 主執行腳本
├── run_all.ps1                # PowerShell 執行腳本
├── requirements.txt           # Python 依賴
├── README.md
├── src/
│   ├── data_loader.py         # 資料載入
│   ├── indicators.py          # 技術指標與因子
│   ├── backtester.py          # 回測引擎
│   ├── optimizer.py           # Walk-Forward 優化
│   ├── visualizer.py          # 視覺化模組
│   └── strategies/
│       ├── factor_strategy.py # 因子策略
│       └── route_configs.py   # 路線配置
├── outputs/                   # 輸出結果
└── logs/                      # 執行日誌
```

## 注意事項

1. 首次運行會自動下載 ETH 價格數據（約需 5-10 分鐘）
2. 完整優化 (100 trials x 3 routes) 約需 30-60 分鐘
3. 資料會快取在 `data/` 目錄，後續運行更快
4. 如需重新下載數據，刪除 `data/eth_1h.parquet`

## 風險聲明

本專案僅供研究與教育目的。回測結果不保證未來表現。
實盤交易需考慮更多因素，包括但不限於流動性、執行延遲等。

