import pandas as pd
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
    _ = source_tag
    _ = is_realtime

    # Try fetching via tvDatafeed if available.
    if TvDatafeed:
        try:
            tv = TvDatafeed()
            symbol = ticker
            exchange = "TVC"  # Default for many symbols in this project.

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
            elif "/" in ticker:
                symbol, exchange = ticker.split("/", 1)

            data = (
                tv.get_hist(
                    symbol=symbol,
                    exchange=exchange,
                    interval=Interval.in_1_minute,
                    n_bars=2,
                )
                if Interval
                else None
            )

            if data is not None and len(data) >= 1:
                latest_price = float(data["close"].iloc[-1])
                last_ts = pd.to_datetime(data.index[-1])
                as_of = last_ts.strftime("%Y-%m-%d %H:%M:%S")

                prev_price = float(data["close"].iloc[-2]) if len(data) >= 2 else latest_price
                price_change = latest_price - prev_price
                pct_change = (price_change / prev_price) * 100 if prev_price else 0.0

                return {
                    "ok": True,
                    "latest_price": f"{latest_price:,.2f}",
                    "price_change": f"{price_change:+,.2f}",
                    "pct_change": f"{pct_change:+.2f}%",
                    "is_positive": price_change >= 0,
                    "as_of": as_of,
                    "source": "tvdatafeed",
                }
        except Exception:
            # Silence tvdatafeed errors and fallback to CSV.
            pass

    # Fallback to CSV.
    try:
        path = Path(processed_file_path)
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if not df.empty and "Close" in df.columns:
                latest_price = float(df["Close"].iloc[-1])
                prev_price = float(df["Close"].iloc[-2]) if len(df) >= 2 else latest_price
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
                    "source": "csv_fallback",
                }
    except Exception:
        pass

    return {
        "ok": False,
        "error": "Could not fetch price from live feed or CSV fallback.",
        "latest_price": "N/A",
        "price_change": "N/A",
        "pct_change": "N/A",
        "is_positive": True,
        "as_of": "N/A",
    }
