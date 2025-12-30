"""
Data Loader for ETH-USD with fallback sources.
Supports 2020-01-01 to 2025-11-30 period.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Also check parent ETH BackTest data
PARENT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "ETH BackTest" / "data"


def get_cache_path(freq: str) -> Path:
    """Get cache file path for given frequency."""
    return DATA_DIR / f"eth_{freq}.parquet"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase."""
    df.columns = [c.lower() for c in df.columns]
    
    rename_map = {
        'date': 'timestamp',
        'datetime': 'timestamp',
        'time': 'timestamp',
        'adj close': 'close',
        'adj_close': 'close',
    }
    
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    if 'timestamp' not in df.columns:
        df = df.reset_index()
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'timestamp'})
        elif 'Date' in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})
    
    df = df[[c for c in required if c in df.columns]]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def fetch_yfinance(start_date: str, end_date: str, freq: str = '1h') -> pd.DataFrame:
    """Fetch data from yfinance."""
    import yfinance as yf
    
    interval_map = {
        '1h': '1h',
        '60m': '1h',
        '1d': '1d',
        '4h': '1h',
    }
    interval = interval_map.get(freq, '1h')
    
    print(f"Fetching ETH-USD from yfinance (interval={interval})...")
    
    ticker = yf.Ticker("ETH-USD")
    
    if interval == '1h':
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        chunks = []
        current_start = start
        chunk_days = 700
        
        while current_start < end:
            current_end = min(current_start + timedelta(days=chunk_days), end)
            
            try:
                chunk = ticker.history(
                    start=current_start.strftime('%Y-%m-%d'),
                    end=current_end.strftime('%Y-%m-%d'),
                    interval=interval
                )
                if not chunk.empty:
                    chunks.append(chunk)
                    print(f"  Fetched {len(chunk)} rows from {current_start.date()} to {current_end.date()}")
            except Exception as e:
                print(f"  Warning: Failed to fetch chunk {current_start.date()} to {current_end.date()}: {e}")
            
            current_start = current_end
        
        if chunks:
            df = pd.concat(chunks)
        else:
            df = pd.DataFrame()
    else:
        df = ticker.history(start=start_date, end=end_date, interval=interval)
    
    if df.empty:
        raise ValueError("yfinance returned empty data")
    
    df = normalize_columns(df)
    
    if freq == '4h':
        df = df.set_index('timestamp')
        df = df.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
    
    return df


def fetch_cryptocompare(start_date: str, end_date: str, freq: str = '1h') -> pd.DataFrame:
    """Fetch data from CryptoCompare (free endpoint)."""
    import requests
    
    print("Fetching ETH-USD from CryptoCompare...")
    
    if freq in ['1h', '60m']:
        endpoint = 'histohour'
        limit_per_request = 2000
    elif freq == '1d':
        endpoint = 'histoday'
        limit_per_request = 2000
    else:
        endpoint = 'histohour'
        limit_per_request = 2000
    
    base_url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    all_data = []
    to_ts = int(end.timestamp())
    
    while True:
        params = {
            'fsym': 'ETH',
            'tsym': 'USD',
            'limit': limit_per_request,
            'toTs': to_ts
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            data = response.json()
            
            if data.get('Response') != 'Success':
                print(f"  CryptoCompare error: {data.get('Message', 'Unknown error')}")
                break
            
            records = data.get('Data', {}).get('Data', [])
            if not records:
                break
            
            all_data.extend(records)
            
            oldest_ts = min(r['time'] for r in records)
            if oldest_ts <= start.timestamp():
                break
            
            to_ts = oldest_ts - 1
            print(f"  Fetched up to {datetime.fromtimestamp(oldest_ts).date()}, total: {len(all_data)} rows")
            
        except Exception as e:
            print(f"  Error fetching from CryptoCompare: {e}")
            break
    
    if not all_data:
        raise ValueError("CryptoCompare returned empty data")
    
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'volumeto': 'volume'})
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def load_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-11-30",
    freq: str = "1h",
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Load ETH-USD data with caching and fallback sources.
    """
    cache_path = get_cache_path(freq)
    parent_cache = PARENT_DATA_DIR / f"eth_{freq}.parquet"
    
    # Try to load from parent cache first
    if parent_cache.exists() and not force_refresh:
        print(f"Loading from parent cache: {parent_cache}")
        df = pd.read_parquet(parent_cache)
        df = normalize_columns(df)
        
        cache_start = df['timestamp'].min()
        cache_end = df['timestamp'].max()
        req_start = pd.to_datetime(start_date)
        req_end = pd.to_datetime(end_date)
        
        if cache_start <= req_start and cache_end >= req_end - timedelta(days=1):
            df = df[(df['timestamp'] >= req_start) & (df['timestamp'] <= req_end)]
            print(f"Using parent cache. Rows: {len(df)}, Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            return df.reset_index(drop=True)
    
    # Try local cache
    if cache_path.exists() and not force_refresh:
        print(f"Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)
        
        cache_start = df['timestamp'].min()
        cache_end = df['timestamp'].max()
        req_start = pd.to_datetime(start_date)
        req_end = pd.to_datetime(end_date)
        
        if cache_start <= req_start and cache_end >= req_end - timedelta(days=1):
            print(f"Cache covers requested range. Using cached data.")
            df = df[(df['timestamp'] >= req_start) & (df['timestamp'] <= req_end)]
            return df.reset_index(drop=True)
        else:
            print(f"Cache does not cover full range. Refreshing...")
    
    # Fetch from sources
    df = None
    
    try:
        df = fetch_yfinance(start_date, end_date, freq)
        print(f"Successfully fetched {len(df)} rows from yfinance")
    except Exception as e:
        print(f"yfinance failed: {e}")
    
    if df is None or df.empty:
        try:
            df = fetch_cryptocompare(start_date, end_date, freq)
            print(f"Successfully fetched {len(df)} rows from CryptoCompare")
        except Exception as e:
            print(f"CryptoCompare failed: {e}")
    
    if df is None or df.empty:
        raise ValueError("Failed to fetch data from all sources")
    
    print(f"Saving data to cache: {cache_path}")
    df.to_parquet(cache_path, index=False)
    
    return df


def resample_to_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly data to daily."""
    df = df_hourly.copy()
    df = df.set_index('timestamp')
    df_daily = df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    return df_daily


def get_data_info(df: pd.DataFrame) -> dict:
    """Get summary info about the data."""
    return {
        'rows': len(df),
        'start_date': df['timestamp'].min().strftime('%Y-%m-%d %H:%M'),
        'end_date': df['timestamp'].max().strftime('%Y-%m-%d %H:%M'),
        'missing_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
        'price_range': (float(df['close'].min()), float(df['close'].max())),
    }


if __name__ == "__main__":
    df = load_data(start_date="2020-01-01", end_date="2025-11-30", freq="1h")
    print("\nData Info:")
    print(get_data_info(df))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())

