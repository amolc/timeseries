import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_landing_assets_data():
    """
    Fetches 3-month data for tracked assets.
    Returns a dictionary of dataframes.
    """
    assets = {
        'BTCUSD': 'BTC-USD',
        'PAXUSD': 'PAXG-USD',
        'SPX500': '^GSPC',
        'GOLD': 'GC=F',
        'NIFTY': '^NSEI',
        'USOIL': 'CL=F',
    }
    
    data_results = {}
    
    for name, ticker in assets.items():
        try:
            # Fetch 3 months of daily data
            df = yf.download(ticker, period="3mo", interval="1d")
            if df is not None and not df.empty:
                # Handle MultiIndex columns (common in newer yfinance versions)
                if isinstance(df.columns, pd.MultiIndex):
                    # Try to find the level that contains 'Close'
                    if 'Close' in df.columns.get_level_values(0):
                        df.columns = df.columns.get_level_values(0)
                    else:
                        df.columns = df.columns.get_level_values(1)
                data_results[name] = df
            else:
                data_results[name] = None
        except Exception as e:
            print(f"Error fetching data for {name}: {e}")
            data_results[name] = None
            
    return data_results
