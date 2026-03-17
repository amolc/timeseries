import sys
sys.path.append("d:/PranitCode/timeseries")
import pandas as pd
from spx500.data.ingestion import download_spx500_data

print("Downloading via TVDatafeed...")
try:
    df = download_spx500_data("1h", n_bars=10)
    print("DataFrame Info:")
    print(df.info())
    print("\nTail:")
    print(df.tail())
except Exception as e:
    print("Error:", e)
