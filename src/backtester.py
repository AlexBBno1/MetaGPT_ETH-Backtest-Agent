"""
Backtesting Engine for ETH Empirical Routes 5Y.
Supports long/short positions, leverage, transaction costs.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    initial_capital: float = 15000.0  # 15K USD as specified
    commission_bps: float = 4.0       # 4 bps per trade
    slippage_bps: float = 5.0         # 5 bps per trade
    leverage: float = 1.0
    max_leverage: float = 2.5
    execution_price: str = 'next_open'  # 'next_open' or 'close'
    
    # Risk management
    max_drawdown_stop: Optional[float] = None  # Optional hard stop
    
    # Position limits
    max_position: float = 1.0
    min_position: float = -1.0


@dataclass
class BacktestResult:
    """Results from backtest."""
    metrics: Dict
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    config: BacktestConfig


def calculate_metrics(
    equity: pd.Series,
    returns: pd.Series,
    trades_df: pd.DataFrame,
    periods_per_year: int = 24 * 365,  # hourly data
    risk_free_rate: float = 0.0
) -> Dict:
    """
    Calculate performance metrics.
    """
    equity = equity.dropna()
    returns = returns.dropna()
    
    if len(equity) < 2:
        return {k: 0.0 for k in ['CAGR', 'annual_vol', 'Sharpe', 'Sortino', 
                                  'MaxDrawdown', 'Calmar', 'WinRate', 'ProfitFactor',
                                  'TradesCount', 'AvgTradeReturn', 'Exposure', 'Turnover']}
    
    # Time period in years
    n_periods = len(equity)
    years = n_periods / periods_per_year
    
    # CAGR
    total_return = equity.iloc[-1] / equity.iloc[0]
    if years > 0 and total_return > 0:
        cagr = (total_return ** (1 / years)) - 1
    else:
        cagr = 0.0
    
    # Annual volatility
    annual_vol = returns.std() * np.sqrt(periods_per_year)
    
    # Sharpe Ratio
    excess_returns = returns - risk_free_rate / periods_per_year
    if returns.std() > 0:
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0
    
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)
    else:
        sortino = 0.0
    
    # Maximum Drawdown
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())
    
    # Calmar Ratio
    if max_drawdown > 0:
        calmar = cagr / max_drawdown
    else:
        calmar = 0.0
    
    # Trade statistics
    trades_count = len(trades_df)
    
    if trades_count > 0:
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / trades_count
        
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        
        if total_loss > 0:
            profit_factor = total_profit / total_loss
        else:
            profit_factor = float('inf') if total_profit > 0 else 0.0
        
        avg_trade_return = trades_df['return'].mean() if 'return' in trades_df.columns else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_return = 0.0
    
    exposure = 0.5
    turnover = 0.0
    
    return {
        'CAGR': round(cagr * 100, 2),
        'annual_vol': round(annual_vol * 100, 2),
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'MaxDrawdown': round(max_drawdown * 100, 2),
        'Calmar': round(calmar, 2),
        'WinRate': round(win_rate * 100, 2),
        'ProfitFactor': round(min(profit_factor, 100), 2),
        'TradesCount': trades_count,
        'AvgTradeReturn': round(avg_trade_return * 100, 4),
        'Exposure': round(exposure * 100, 2),
        'Turnover': round(turnover * 100, 2),
        'TotalReturn': round((total_return - 1) * 100, 2),
        'FinalEquity': round(equity.iloc[-1], 2),
        'Years': round(years, 4),
    }


class Backtester:
    """Vectorized backtesting engine supporting long/short positions."""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
    
    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        periods_per_year: int = 24 * 365
    ) -> BacktestResult:
        """
        Run backtest on given data and signals.
        """
        df = df.copy()
        signals = signals.copy()
        
        # Align signals with data
        signals = signals.reindex(df.index).fillna(0)
        
        # Clip positions
        signals = signals.clip(self.config.min_position, self.config.max_position)
        
        # Apply leverage
        signals = signals * self.config.leverage
        signals = signals.clip(-self.config.max_leverage, self.config.max_leverage)
        
        # Execution price
        if self.config.execution_price == 'next_open':
            exec_price = df['open'].shift(-1).fillna(df['close'])
        else:
            exec_price = df['close']
        
        # Calculate position changes
        position = signals.shift(1).fillna(0)
        position_change = position.diff().fillna(0)
        
        # Returns (based on close-to-close)
        price_return = df['close'].pct_change().fillna(0)
        
        # Strategy returns (before costs)
        strategy_return_gross = position * price_return
        
        # Transaction costs
        total_cost_bps = self.config.commission_bps + self.config.slippage_bps
        cost_fraction = total_cost_bps / 10000
        
        # Cost is proportional to absolute position change
        transaction_cost = abs(position_change) * cost_fraction
        
        # Net returns
        strategy_return_net = strategy_return_gross - transaction_cost
        
        # Build equity curve
        equity = (1 + strategy_return_net).cumprod() * self.config.initial_capital
        
        # Generate trade log
        trades_df = self._generate_trades(df, position, exec_price, transaction_cost)
        
        # Calculate exposure and turnover (turnover per spec = sum |Δ position| annualized)
        exposure = (position.abs() > 0.01).mean()
        years = max(len(df) / periods_per_year, 1e-9)
        turnover = abs(position_change).sum() / years
        
        # Calculate metrics
        metrics = calculate_metrics(
            equity=equity,
            returns=strategy_return_net,
            trades_df=trades_df,
            periods_per_year=periods_per_year
        )
        metrics['Exposure'] = round(exposure * 100, 2)
        metrics['Turnover'] = round(turnover, 2)
        metrics['Years'] = round(years, 4)
        metrics['TradesPerYear'] = round(len(trades_df) / years if years > 0 else 0, 2)
        
        # Build equity curve DataFrame
        equity_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'equity': equity.values,
            'position': position.values,
            'returns': strategy_return_net.values,
            'drawdown': ((equity - equity.expanding().max()) / equity.expanding().max()).values,
            'cumulative_return': ((equity / self.config.initial_capital) - 1).values,
        })
        
        # Signals DataFrame
        signals_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'close': df['close'],
            'signal': signals.values,
            'position': position.values,
        })
        
        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_df,
            trades=trades_df,
            signals=signals_df,
            config=self.config
        )
    
    def _generate_trades(
        self,
        df: pd.DataFrame,
        position: pd.Series,
        exec_price: pd.Series,
        transaction_cost: pd.Series
    ) -> pd.DataFrame:
        """Generate trade log from position series."""
        trades = []
        
        current_trade = None
        prev_pos = 0
        
        for i in range(len(position)):
            pos = position.iloc[i]
            
            if abs(pos - prev_pos) > 0.001:
                timestamp = df['timestamp'].iloc[i]
                price = exec_price.iloc[i]
                cost = transaction_cost.iloc[i] * self.config.initial_capital
                
                if current_trade is not None and (np.sign(pos) != np.sign(prev_pos) or abs(pos) < 0.001):
                    current_trade['exit_time'] = timestamp
                    current_trade['exit_price'] = price
                    current_trade['exit_cost'] = cost
                    
                    if current_trade['side'] == 'long':
                        pnl = (current_trade['exit_price'] - current_trade['entry_price']) * current_trade['size']
                    else:
                        pnl = (current_trade['entry_price'] - current_trade['exit_price']) * current_trade['size']
                    
                    pnl -= current_trade['entry_cost'] + current_trade['exit_cost']
                    current_trade['pnl'] = pnl
                    current_trade['return'] = pnl / (current_trade['entry_price'] * current_trade['size']) if current_trade['size'] > 0 else 0
                    
                    holding = (current_trade['exit_time'] - current_trade['entry_time']).total_seconds() / 86400
                    current_trade['holding_days'] = holding
                    current_trade['fees'] = current_trade['entry_cost'] + current_trade['exit_cost']
                    
                    trades.append(current_trade)
                    current_trade = None
                
                if abs(pos) > 0.001:
                    side = 'long' if pos > 0 else 'short'
                    size = abs(pos) * self.config.initial_capital / price
                    
                    current_trade = {
                        'entry_time': timestamp,
                        'side': side,
                        'entry_price': price,
                        'size': size,
                        'entry_cost': cost,
                    }
            
            prev_pos = pos
        
        # Close any remaining trade
        if current_trade is not None:
            current_trade['exit_time'] = df['timestamp'].iloc[-1]
            current_trade['exit_price'] = df['close'].iloc[-1]
            current_trade['exit_cost'] = 0
            
            if current_trade['side'] == 'long':
                pnl = (current_trade['exit_price'] - current_trade['entry_price']) * current_trade['size']
            else:
                pnl = (current_trade['entry_price'] - current_trade['exit_price']) * current_trade['size']
            
            pnl -= current_trade['entry_cost']
            current_trade['pnl'] = pnl
            current_trade['return'] = pnl / (current_trade['entry_price'] * current_trade['size']) if current_trade['size'] > 0 else 0
            current_trade['holding_days'] = (current_trade['exit_time'] - current_trade['entry_time']).total_seconds() / 86400
            current_trade['fees'] = current_trade['entry_cost']
            
            trades.append(current_trade)
        
        if trades:
            trades_df = pd.DataFrame(trades)
            cols = ['entry_time', 'exit_time', 'side', 'entry_price', 'exit_price', 
                    'size', 'pnl', 'return', 'holding_days', 'fees']
            cols = [c for c in cols if c in trades_df.columns]
            return trades_df[cols]
        else:
            return pd.DataFrame(columns=['entry_time', 'exit_time', 'side', 'entry_price', 
                                          'exit_price', 'size', 'pnl', 'return', 'holding_days', 'fees'])


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 15000.0,
    commission_bps: float = 4.0,
    slippage_bps: float = 5.0,
    leverage: float = 1.0,
    periods_per_year: int = 24 * 365
) -> BacktestResult:
    """Convenience function to run backtest with common parameters."""
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        leverage=leverage
    )
    
    backtester = Backtester(config)
    return backtester.run(df, signals, periods_per_year)


if __name__ == "__main__":
    from data_loader import load_data
    
    df = load_data(start_date="2023-01-01", end_date="2024-01-01", freq="1h")
    
    # Simple momentum signal for testing
    signals = pd.Series(np.sign(df['close'].pct_change(24)), index=df.index)
    
    result = run_backtest(df, signals)
    
    print("Metrics:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")
    
    print(f"\nTrades: {len(result.trades)}")
    if len(result.trades) > 0:
        print(result.trades.head())

