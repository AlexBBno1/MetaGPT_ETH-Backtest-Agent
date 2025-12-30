"""
Technical Indicators with Factor Exploration Results.
Implements all validated factors:
- Trend Persistence (ADX + Hurst)
- Chop Detector
- Rules-Based Regime
- Volatility Breakout
- Crash Protection
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


# =============================================================================
# Basic Indicators
# =============================================================================

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def donchian(df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channel. Returns (upper, lower, middle)."""
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    return upper, lower, middle


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands. Returns (upper, middle, lower)."""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def keltner_channel(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10,
                    atr_mult: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channel. Returns (upper, middle, lower)."""
    middle = ema(df['close'], ema_period)
    atr_val = atr(df, atr_period)
    upper = middle + atr_mult * atr_val
    lower = middle - atr_mult * atr_val
    return upper, middle, lower


def realized_volatility(series: pd.Series, period: int = 20, annualize: bool = True,
                        periods_per_year: int = 24 * 365) -> pd.Series:
    """Realized volatility from log returns."""
    log_ret = np.log(series / series.shift(1))
    vol = log_ret.rolling(window=period).std()
    
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    
    return vol


# =============================================================================
# Factor F1: Trend Persistence (ADX + Hurst Exponent)
# =============================================================================

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index - Trend Strength."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed indicators
    atr_smooth = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_smooth
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_smooth
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.ewm(span=period, adjust=False).mean()
    
    return adx_val.fillna(0)


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> pd.Series:
    """
    Simplified Hurst Exponent estimation using R/S analysis.
    H > 0.5: Trending (persistent)
    H = 0.5: Random walk
    H < 0.5: Mean reverting (anti-persistent)
    """
    def calc_hurst(window):
        if len(window) < max_lag:
            return 0.5
        
        try:
            lags = range(2, min(max_lag, len(window) // 2))
            if len(list(lags)) < 2:
                return 0.5
            
            # Calculate R/S for each lag
            rs_values = []
            for lag in lags:
                # Calculate mean-adjusted returns
                returns = np.diff(window[:lag+1])
                if len(returns) == 0:
                    continue
                    
                mean_ret = np.mean(returns)
                deviations = np.cumsum(returns - mean_ret)
                
                R = np.max(deviations) - np.min(deviations)
                S = np.std(returns, ddof=1)
                
                if S > 0 and R > 0:
                    rs_values.append((lag, R / S))
            
            if len(rs_values) < 2:
                return 0.5
            
            # Linear regression in log-log space
            lags_arr = np.array([x[0] for x in rs_values])
            rs_arr = np.array([x[1] for x in rs_values])
            
            log_lags = np.log(lags_arr)
            log_rs = np.log(rs_arr)
            
            # Simple linear regression
            slope = np.polyfit(log_lags, log_rs, 1)[0]
            
            return np.clip(slope, 0, 1)
        except:
            return 0.5
    
    # Rolling Hurst with minimum 50 periods
    hurst = series.rolling(window=max_lag * 3, min_periods=max_lag * 2).apply(calc_hurst, raw=True)
    return hurst.fillna(0.5)


def trend_persistence(df: pd.DataFrame, adx_period: int = 14, hurst_period: int = 20,
                      adx_threshold: float = 25.0) -> pd.Series:
    """
    Trend Persistence Factor combining ADX and Hurst.
    Returns: Score from 0 (no trend) to 1 (strong trend)
    """
    adx_val = adx(df, adx_period)
    hurst_val = hurst_exponent(df['close'], hurst_period)
    
    # ADX component: 0 to 1 (normalize with threshold at 50)
    adx_score = (adx_val / 50).clip(0, 1)
    
    # Hurst component: 0.5+ means trending
    hurst_score = ((hurst_val - 0.5) * 2).clip(0, 1)
    
    # Combined score (weighted average)
    trend_score = 0.6 * adx_score + 0.4 * hurst_score
    
    return trend_score


# =============================================================================
# Factor F1: Chop Detector
# =============================================================================

def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Choppiness Index: 0-100 scale
    High values (>60) = Choppy/ranging market
    Low values (<40) = Trending market
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over period
    atr_sum = tr.rolling(window=period).sum()
    
    # Highest high and lowest low over period
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop.fillna(50)


def chop_detector(df: pd.DataFrame, period: int = 14, chop_threshold: float = 60.0) -> pd.Series:
    """
    Chop Detector: Returns 1 if market is choppy (should avoid trading)
    Returns 0 if market is trending (good for trading)
    """
    chop = choppiness_index(df, period)
    is_choppy = (chop > chop_threshold).astype(float)
    return is_choppy


# =============================================================================
# Factor F2: Rules-Based Regime
# =============================================================================

def rules_regime(df: pd.DataFrame, adx_period: int = 14, sma_period: int = 50,
                 atr_period: int = 14) -> pd.Series:
    """
    Rules-Based Regime Classification:
    0 = RANGING (ADX < 20)
    1 = TRENDING_UP (ADX > 25 & Close > SMA)
    -1 = TRENDING_DOWN (ADX > 25 & Close < SMA)
    2 = VOLATILE (ATR percentile > 80)
    """
    adx_val = adx(df, adx_period)
    sma_val = sma(df['close'], sma_period)
    atr_val = atr(df, atr_period)
    
    # ATR percentile
    atr_pct = atr_val.rolling(window=100, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )
    
    regime = pd.Series(0, index=df.index)  # Default: RANGING
    
    # TRENDING_UP
    trending_up = (adx_val > 25) & (df['close'] > sma_val)
    regime[trending_up] = 1
    
    # TRENDING_DOWN
    trending_down = (adx_val > 25) & (df['close'] < sma_val)
    regime[trending_down] = -1
    
    # VOLATILE (overrides other states)
    volatile = atr_pct > 0.8
    regime[volatile] = 2
    
    return regime


def regime_position_sizing(regime: pd.Series, base_position: pd.Series,
                          regime_multipliers: dict = None) -> pd.Series:
    """
    Adjust position size based on regime.
    """
    if regime_multipliers is None:
        regime_multipliers = {
            0: 0.5,   # RANGING: half size
            1: 1.2,   # TRENDING_UP: full+ size
            -1: 1.2,  # TRENDING_DOWN: full+ size
            2: 0.7,   # VOLATILE: reduced size
        }
    
    multipliers = regime.map(regime_multipliers).fillna(1.0)
    return base_position * multipliers


# =============================================================================
# Factor F3: Volatility Breakout
# =============================================================================

def bollinger_squeeze(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0) -> pd.Series:
    """
    Detect Bollinger Band squeeze (band width at 20-day low).
    Returns True when squeeze is on.
    """
    bb_upper, bb_mid, bb_lower = bollinger_bands(df['close'], bb_period, bb_std)
    bb_width = (bb_upper - bb_lower) / bb_mid
    
    bb_width_min = bb_width.rolling(window=20).min()
    squeeze_on = bb_width <= bb_width_min * 1.05  # Within 5% of 20-day low
    
    return squeeze_on


def volatility_breakout_signal(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
                               atr_period: int = 14) -> pd.Series:
    """
    Volatility Breakout Strategy Signal.
    Trigger: BB width at 20-day low, then price breaks upper/lower band.
    Returns: 1 (long), -1 (short), 0 (no signal)
    """
    bb_upper, bb_mid, bb_lower = bollinger_bands(df['close'], bb_period, bb_std)
    squeeze = bollinger_squeeze(df, bb_period, bb_std)
    
    # Track if we were in squeeze recently (last 3 bars)
    squeeze_recent = squeeze.rolling(window=3).max().fillna(0) > 0
    
    # Breakout signals
    breakout_up = (df['close'] > bb_upper) & squeeze_recent.shift(1)
    breakout_down = (df['close'] < bb_lower) & squeeze_recent.shift(1)
    
    signal = pd.Series(0, index=df.index)
    signal[breakout_up] = 1
    signal[breakout_down] = -1
    
    return signal


# =============================================================================
# Factor F3: Crash Protection
# =============================================================================

def momentum_factor(df: pd.DataFrame, lookback: int = 24) -> pd.Series:
    """Price momentum over lookback period."""
    return df['close'].pct_change(lookback)


def crash_protection_signal(df: pd.DataFrame, vol_lookback: int = 20, 
                            crash_vol_mult: float = 2.0,
                            momentum_lookback: int = 24) -> pd.Series:
    """
    Crash Protection: Detect momentum crashes and reduce exposure.
    Returns: Multiplier 0-1 (1 = full exposure, 0 = no exposure)
    """
    # Realized volatility
    vol = realized_volatility(df['close'], vol_lookback, annualize=False)
    vol_ma = vol.rolling(window=vol_lookback * 2).mean()
    
    # High volatility regime
    high_vol = vol > vol_ma * crash_vol_mult
    
    # Momentum crash detection
    momentum = momentum_factor(df, momentum_lookback)
    momentum_ma = momentum.rolling(window=momentum_lookback).mean()
    
    # Crash: Sharp negative momentum + high volatility
    crash_signal = (momentum < -0.05) & high_vol
    
    # Protection multiplier
    protection = pd.Series(1.0, index=df.index)
    protection[high_vol] = 0.7  # Reduce exposure in high vol
    protection[crash_signal] = 0.3  # Significantly reduce in crash
    
    # Smooth the protection signal
    protection = protection.rolling(window=3).mean().fillna(1.0)
    
    return protection


# =============================================================================
# Combined Factor Calculations
# =============================================================================

def calculate_all_factors(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    Calculate all factors for the factor-based strategy.
    """
    if params is None:
        params = {}
    
    df = df.copy()
    
    # Basic indicators
    df['sma_20'] = sma(df['close'], 20)
    df['sma_50'] = sma(df['close'], 50)
    df['ema_12'] = ema(df['close'], 12)
    df['ema_26'] = ema(df['close'], 26)
    df['ema_50'] = ema(df['close'], 50)
    df['rsi'] = rsi(df['close'], params.get('rsi_period', 14))
    df['atr'] = atr(df, params.get('atr_period', 14))
    df['atr_pct'] = df['atr'] / df['close'] * 100
    
    # Donchian
    don_period = params.get('donchian_period', 20)
    df['don_upper'], df['don_lower'], df['don_mid'] = donchian(df, don_period)
    
    # Bollinger Bands
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = bollinger_bands(df['close'], bb_period, bb_std)
    
    # Realized volatility
    df['realized_vol'] = realized_volatility(df['close'], 20)
    
    # Factor F1: Trend Persistence
    df['adx'] = adx(df, params.get('adx_period', 14))
    df['trend_persistence'] = trend_persistence(
        df, 
        adx_period=params.get('adx_period', 14),
        hurst_period=params.get('hurst_period', 20),
        adx_threshold=params.get('adx_threshold', 25)
    )
    
    # Factor F1: Chop Detector
    df['chop_index'] = choppiness_index(df, params.get('chop_period', 14))
    df['is_choppy'] = chop_detector(
        df, 
        period=params.get('chop_period', 14),
        chop_threshold=params.get('chop_threshold', 60)
    )
    
    # Factor F2: Rules-Based Regime
    df['regime'] = rules_regime(
        df,
        adx_period=params.get('regime_adx_period', 14),
        sma_period=params.get('regime_sma_period', 50),
        atr_period=params.get('regime_atr_period', 14)
    )
    
    # Factor F3: Volatility Breakout
    df['vol_breakout_signal'] = volatility_breakout_signal(
        df,
        bb_period=params.get('vol_bb_period', 20),
        bb_std=params.get('vol_bb_std', 2.0),
        atr_period=params.get('vol_atr_period', 14)
    )
    df['bb_squeeze'] = bollinger_squeeze(df, bb_period, bb_std)
    
    # Factor F3: Crash Protection
    df['crash_protection'] = crash_protection_signal(
        df,
        vol_lookback=params.get('crash_vol_lookback', 20),
        crash_vol_mult=params.get('crash_vol_mult', 2.0),
        momentum_lookback=params.get('crash_momentum_lookback', 24)
    )
    
    # Momentum
    df['momentum_24'] = df['close'].pct_change(24)
    df['momentum_72'] = df['close'].pct_change(72)
    
    return df


if __name__ == "__main__":
    from data_loader import load_data
    
    df = load_data(start_date="2023-01-01", end_date="2024-01-01", freq="1h")
    df = calculate_all_factors(df)
    
    print("Columns:", df.columns.tolist())
    print("\nSample data (factor columns):")
    factor_cols = ['timestamp', 'close', 'adx', 'trend_persistence', 'is_choppy', 'regime', 'vol_breakout_signal', 'crash_protection']
    print(df[factor_cols].tail(20))

