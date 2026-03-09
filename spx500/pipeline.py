import os
import sys
import argparse
import pathlib

# Path setup
CURR_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURR_DIR.parent

if str(CURR_DIR) not in sys.path:
    sys.path.insert(0, str(CURR_DIR))

from data.ingestion import collect_all_intervals
from models.linear_regression import train_linear_regression
from models.arima_model import train_arima_model

def run_full_pipeline(skip_ingestion=False, models=("lr", "arima"), interval_filter=None):
    """
    Main pipeline entry point for SPX500 intervals (1h, 1d, 1w, 1m).
    If interval_filter is provided, only that interval is processed.
    """
    intervals = [interval_filter] if interval_filter else ["1h", "1d", "1w", "1m"]

    if not skip_ingestion:
        print(f"Starting SPX500 data ingestion for intervals: {intervals}...")
        results = collect_all_intervals()
        if not results:
            print("Ingestion failed to produce data.")
            return
    
    pipeline_results = {}
    
    for interval in intervals:
        print(f"\n--- Running models for SPX500 {interval} interval ---")
        interval_results = {}
        
        if "lr" in models:
            lr_out = train_linear_regression(interval=interval)
            if lr_out:
                model, mse, mae, pred_next = lr_out
                interval_results["linear_regression"] = {"mse": mse, "mae": mae, "prediction": pred_next}
        
        if "arima" in models:
            arima_out = train_arima_model(interval=interval)
            if arima_out:
                model_fit, mse, pred_next = arima_out
                interval_results["arima"] = {"mse": mse, "prediction": pred_next}
        
        pipeline_results[interval] = interval_results
    
    print(f"\nSPX500 Pipeline execution complete for {intervals}.")
    return pipeline_results

def parse_args():
    p = argparse.ArgumentParser(description="End-to-end multi-interval SPX500 pipeline")
    p.add_argument("--skip-ingestion", action="store_true", help="Skip data download/preprocess")
    p.add_argument("--models", type=str, default="lr,arima", help="Comma-separated: lr, arima")
    p.add_argument("--interval", type=str, help="Specific interval to run (1h, 1d, 1w, 1m)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    run_full_pipeline(skip_ingestion=args.skip_ingestion, models=models, interval_filter=args.interval)
