import yfinance as yf
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas_ta as ta

# Define directory structure relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def download_spx500_data(interval="1h", period="60d", n_bars=2000):
    """
    Downloads historical SPX500 data from OANDA via TVDataFeed.
    Symbol: SPX500USD
    Exchange: OANDA
    Supports 24h market data for the index.
    """
    symbol = "SPX500USD"
    exchange = "OANDA"
    
    print(f"Downloading {interval} data for {symbol} from {exchange}...")
    
    try:
        from tvDatafeed import TvDatafeed, Interval
        tv = TvDatafeed()
        
        # Map interval string to Interval enum
        interval_map = {
            "1h": Interval.in_1_hour,
            "1d": Interval.in_daily,
            "1w": Interval.in_weekly,
            "1m": Interval.in_monthly
        }
        
        tv_interval = interval_map.get(interval)
        if not tv_interval:
            raise ValueError(f"Unsupported interval: {interval}")
            
        data = tv.get_hist(symbol=symbol, exchange=exchange, interval=tv_interval, n_bars=n_bars)
        
        if data is None or data.empty:
            print(f"tvdatafeed returned empty data for {interval}. Falling back to Yahoo Finance...")
            return download_spx500_data_yfinance(interval=interval, period=period)
            
        # Standardize columns
        data.columns = [col.capitalize() for col in data.columns]
        
        os.makedirs(str(RAW_DIR), exist_ok=True)
        file_path = RAW_DIR / f"spx500_{interval}_raw.csv"
        data.to_csv(str(file_path))
        print(f"Raw {interval} data saved to {file_path}")
        return data
        
    except Exception as e:
        print(f"Error with tvdatafeed for {interval}: {e}. Falling back to Yahoo Finance...")
        return download_spx500_data_yfinance(interval=interval, period=period)

def download_spx500_data_yfinance(interval="1h", period="60d"):
    """
    Fallback: Downloads historical ^GSPC data from Yahoo Finance.
    Note: ^GSPC is only active during market hours, unlike OANDA SPX500USD.
    """
    symbol = "^GSPC"
    # Yahoo Finance interval mapping
    yf_interval_map = {
        "1h": "1h",
        "1d": "1d",
        "1w": "1wk",
        "1m": "1mo"
    }
    
    yf_interval = yf_interval_map.get(interval, "1h")
    
    # Adjust period for larger intervals if needed
    if interval == "1w": period = "2y"
    if interval == "1m": period = "max"
    
    print(f"Downloading fallback {interval} data from Yahoo Finance for {symbol}...")
    try:
        data = yf.download(symbol, period=period, interval=yf_interval)
        if data is None or data.empty:
            raise ValueError(f"Yahoo Finance {interval} data is empty.")
            
        if isinstance(data.columns, pd.MultiIndex):
            # For yfinance MultiIndex, we want the price columns (Open, High, Low, Close, Volume)
            # Check if 'Close' is in the columns
            if 'Close' in data.columns:
                data = data['Close'] if isinstance(data['Close'], pd.Series) else data
            else:
                data.columns = data.columns.get_level_values(0)
            
        os.makedirs(str(RAW_DIR), exist_ok=True)
        file_path = RAW_DIR / f"spx500_{interval}_raw.csv"
        data.to_csv(str(file_path))
        return data
    except Exception as e:
        print(f"Error downloading fallback {interval} data: {e}")
        return None

def preprocess_data(df, interval="1h"):
    """
    Enhanced preprocessing using pandas_ta for robust technical indicators.
    """
    if df is None or df.empty:
        return None
        
    df = df.copy()
    
    # Ensure columns are properly named
    # Handle MultiIndex or tuple columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Map columns to standard names
    cols_map = {col.lower(): col for col in df.columns if isinstance(col, str)}
    
    if 'close' in cols_map:
        df['Close'] = df[cols_map['close']]
    if 'open' in cols_map:
        df['Open'] = df[cols_map['open']]
    if 'high' in cols_map:
        df['High'] = df[cols_map['high']]
    if 'low' in cols_map:
        df['Low'] = df[cols_map['low']]
    if 'volume' in cols_map:
        df['Volume'] = df[cols_map['volume']]

    # Add Technical Indicators using pandas_ta
    # 1. RSI
    df.ta.rsi(length=14, append=True)
    
    # 2. Moving Averages
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.ema(length=20, append=True)
    
    # 3. MACD
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # 4. Bollinger Bands
    df.ta.bbands(length=20, std=2, append=True)
    
    # 5. ATR (Volatility)
    df.ta.atr(length=14, append=True)
    
    # Handle NaNs from indicators
    df = df.dropna()
    
    # Create target variable (next interval close)
    df['Target_Close'] = df['Close'].shift(-1)
    
    # Drop the last row as it won't have a target
    df = df.dropna()
    
    os.makedirs(str(PROCESSED_DIR), exist_ok=True)
    file_path = PROCESSED_DIR / f"spx500_{interval}_processed.csv"
    df.to_csv(str(file_path))
    print(f"Processed {interval} data saved to {file_path}")
    
    return df

def collect_all_intervals():
    """
    Utility to collect data for all intervals.
    """
    intervals = ["1h", "1d", "1w", "1m"]
    results = {}
    for interval in intervals:
        raw_data = download_spx500_data(interval=interval)
        if raw_data is not None:
            processed_data = preprocess_data(raw_data, interval=interval)
            results[interval] = processed_data
    return results

if __name__ == "__main__":
    collect_all_intervals()
