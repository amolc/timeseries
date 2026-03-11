import pandas as pd
from datetime import datetime
import os
from pathlib import Path

try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    TvDatafeed = None
    Interval = None

def get_last_price_payload(source_tag, ticker, processed_file_path, is_realtime=False):
    """
    Fetches the latest price for a given ticker using tvdatafeed.
    If tvdatafeed fails or is unavailable, falls back to the last price in the processed CSV file.
    
    Args:
        source_tag (str): Tag for logging source (e.g. 'usoil_dashboard')
        ticker (str): Ticker symbol (e.g. 'USOIL/TVC' or 'BTC-USD')
        processed_file_path (str/Path): Path to the processed CSV file for fallback.
        is_realtime (bool): Unused flag kept for API compatibility.
        
    Returns:
        dict: Payload with price, change, pct_change, is_positive, as_of, and ok flag.
    """
    
    # Try fetching via tvDatafeed if available
    if TvDatafeed:
        try:
            tv = TvDatafeed()
            symbol = ticker
            exchange = 'TVC' # Default for many symbols in this project
            
            # Common ticker mapping for tvdatafeed
            ticker_map = {
                "BTC-USD": ("BTCUSD", "BINANCE"),
                "PAXG-USD": ("PAXGUSD", "BINANCE"),
                "GC=F": ("GOLD", "TVC"),
                "^GSPC": ("SPX500", "TVC"),
                "^NSEI": ("NIFTY", "NSE"),
                "CL=F": ("USOIL", "TVC"),
            }
            
            if ticker in ticker_map:
                symbol, exchange = ticker_map[ticker]
            elif '/' in ticker:
                symbol, exchange = ticker.split('/')
            
            # Special handling for Yahoo Finance tickers in config that might need mapping to TV symbols
            # But for now we use what is passed.
            
            # Use 1 minute for most recent data
            if Interval:
                data = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_minute, n_bars=2)
            else:
                data = None
            
            if data is not None and len(data) >= 1:
                latest_price = float(data['close'].iloc[-1])
                # Ensure index is datetime
                last_ts = pd.to_datetime(data.index[-1])
                as_of = last_ts.strftime("%Y-%m-%d %H:%M:%S")
                
                if len(data) >= 2:
                    prev_price = float(data['close'].iloc[-2])
                else:
                    prev_price = latest_price
                
                price_change = latest_price - prev_price
                pct_change = (price_change / prev_price) * 100 if prev_price else 0.0
                
                return {
                    "ok": True,
                    "latest_price": f"{latest_price:,.2f}",
                    "price_change": f"{price_change:+,.2f}",
                    "pct_change": f"{pct_change:+.2f}%",
                    "is_positive": price_change >= 0,
                    "as_of": as_of,
                    "source": "tvdatafeed"
                }
        except Exception as e:
            # Silence tvdatafeed errors and fallback to CSV
            pass

    # Fallback to CSV
    try:
        path = Path(processed_file_path)
        if path.exists():
            # Use low_memory=False for safety with large files
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if not df.empty:
                # Ensure 'Close' exists
                if 'Close' in df.columns:
                    latest_price = float(df['Close'].iloc[-1])
                    prev_price = float(df['Close'].iloc[-2]) if len(df) >= 2 else latest_price
                    price_change = latest_price - prev_price
                    pct_change = (price_change / prev_price) * 100 if prev_price else 0.0
                    
                    last_ts = pd.to_datetime(df.index[-1])
                    as_of = last_ts.strftime("%Y-%m-%d %H:%M:%S")
                    
                    return {
                        "ok": True,
                        "latest_price": f"{latest_price:,.2f}",
                        "price_change": f"{price_change:+,.2f}",
                        "pct_change": f"{pct_change:+.2f}%",
                        "is_positive": price_change >= 0,
                        "as_of": as_of,
                        "source": "csv_fallback"
                    }
    except Exception as e:
        # Final fallback failure
        pass
        
    return {
        "ok": False,
        "error": "Could not fetch price from live feed or CSV fallback.",
=======
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


_TICKER_ALIAS = {
    "USOIL/TVC": "CL=F",
}


def _empty_payload(reason: str = "unavailable") -> Dict[str, Any]:
    return {
        "ok": False,
        "latest_price": "N/A",
        "price_change": "N/A",
        "pct_change": "N/A",
        "is_positive": True,
        "as_of": "N/A"
        "source": reason,
    }


def _format_payload(latest: float, previous: float, as_of: Any, source: str) -> Dict[str, Any]:
    change = latest - previous if previous else 0.0
    pct_change = (change / previous * 100.0) if previous else 0.0

    try:
        as_of_text = pd.to_datetime(as_of, utc=True).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        as_of_text = str(as_of)

    return {
        "ok": True,
        "latest_price": f"{latest:,.2f}",
        "price_change": f"{change:+,.2f}",
        "pct_change": f"{pct_change:+.2f}%",
        "is_positive": change >= 0,
        "as_of": as_of_text,
        "source": source,
    }


def _normalize_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return None
        close_df = df.xs("Close", axis=1, level=0)
        close_series = close_df.iloc[:, 0] if isinstance(close_df, pd.DataFrame) else close_df
    else:
        if "Close" not in df.columns:
            return None
        close_series = df["Close"]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

    close_series = pd.to_numeric(close_series, errors="coerce").dropna()
    if len(close_series) < 2:
        return None
    return close_series


def _from_yfinance(ticker: str) -> Optional[Dict[str, Any]]:
    if yf is None:
        return None

    symbol = _TICKER_ALIAS.get(ticker, ticker)
    for period, interval in (("2d", "1h"), ("7d", "1d")):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        except Exception:
            continue

        close_series = _normalize_close_series(df)
        if close_series is None:
            continue

        latest = float(close_series.iloc[-1])
        previous = float(close_series.iloc[-2])
        return _format_payload(latest, previous, close_series.index[-1], "yfinance")
    return None


def _from_processed_file(processed_file: Path | str) -> Optional[Dict[str, Any]]:
    path = Path(processed_file)
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None

    close_series = _normalize_close_series(df)
    if close_series is None:
        return None

    latest = float(close_series.iloc[-1])
    previous = float(close_series.iloc[-2])
    return _format_payload(latest, previous, close_series.index[-1], "processed_file")


def get_last_price_payload(
    cache_key: str,
    ticker: str,
    processed_file: Path | str,
    force_live: bool = False,
) -> Dict[str, Any]:
    """Return a consistent payload for last price widgets/APIs."""
    _ = cache_key
    _ = force_live

    payload = _from_yfinance(ticker)
    if payload is not None:
        return payload

    payload = _from_processed_file(processed_file)
    if payload is not None:
        return payload

    return _empty_payload("no_live_or_file_data")
