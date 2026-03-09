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

def download_paxusd_data(interval="1h", period="60d", n_bars=2000):
    """
    Downloads historical PAXGUSDT data. Supports 1h, 1d, 1w, 1m intervals.
    PAXGUSDT is the TradingView symbol for PAX Gold / USDT.
    """
    symbol = "PAXGUSDT"
    exchange = "BINANCE"
    
    print(f"Downloading {interval} data for {symbol}...")
    
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
            return download_paxusd_data_yfinance(interval=interval, period=period)
            
        # Standardize columns
        data.columns = [col.capitalize() for col in data.columns]
        
        os.makedirs(str(RAW_DIR), exist_ok=True)
        file_path = RAW_DIR / f"paxusd_{interval}_raw.csv"
        data.to_csv(str(file_path))
        print(f"Raw {interval} data saved to {file_path}")
        return data
        
    except Exception as e:
        print(f"Error with tvdatafeed for {interval}: {e}. Falling back to Yahoo Finance...")
        return download_paxusd_data_yfinance(interval=interval, period=period)

def download_paxusd_data_yfinance(interval="1h", period="60d"):
    """
    Fallback: Downloads historical PAXG-USD data from Yahoo Finance.
    """
    symbol = "PAXG-USD"
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
            data.columns = data.columns.get_level_values(0)
            
        os.makedirs(str(RAW_DIR), exist_ok=True)
        file_path = RAW_DIR / f"paxusd_{interval}_raw.csv"
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
    
    # Ensure column names are correct for pandas_ta
    df.columns = [col.capitalize() for col in df.columns]
    
    # Fill missing values
    df = df.ffill().bfill()
    
    # Technical Indicators via pandas_ta
    df['MA7'] = ta.sma(df['Close'], length=7)
    df['MA21'] = ta.sma(df['Close'], length=21)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA12'] = ta.ema(df['Close'], length=12)
    df['EMA26'] = ta.ema(df['Close'], length=26)
    
    # Standard deviation for volatility
    df['STD7'] = df['Close'].rolling(window=7).std()
    
    # Returns
    df['Daily_Return'] = ta.percent_return(df['Close'])
    
    # Simple Moving Average for consistency with LR features
    df['SMMA7'] = ta.sma(df['Close'], length=7)
    
    # Drop rows with NaN from calculations
    df = df.dropna()
    
    os.makedirs(str(PROCESSED_DIR), exist_ok=True)
    processed_path = PROCESSED_DIR / f"paxusd_{interval}_processed.csv"
    df.to_csv(str(processed_path))
    print(f"Processed {interval} data saved to {processed_path}")
    return df

def collect_all_intervals():
    """
    Main entry point to collect and process data for all required intervals.
    """
    intervals = {
        "1h": {"period": "60d", "n_bars": 2000},
        "1d": {"period": "2y", "n_bars": 1000},
        "1w": {"period": "5y", "n_bars": 500},
        "1m": {"period": "max", "n_bars": 200}
    }
    
    results = {}
    for interval, params in intervals.items():
        raw_df = download_paxusd_data(interval=interval, period=params['period'], n_bars=params['n_bars'])
        if raw_df is not None:
            processed_df = preprocess_data(raw_df, interval=interval)
            results[interval] = processed_df
            
    return results

if __name__ == "__main__":
    collect_all_intervals()
    print("PAXUSD Multi-interval Data Collection Complete.")
