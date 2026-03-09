from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import mlflow
import os
import pathlib
import warnings

# Suppress ARIMA warnings
warnings.filterwarnings("ignore")

# Path setup
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
# Central MLflow DB
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{MLFLOW_DB_PATH}"

def train_arima_model(interval="1h", p=5, d=1, q=0):
    """
    Trains an ARIMA model on Nifty Close price for a given interval.
    """
    processed_file = PROJECT_ROOT / "nifty" / "data" / "processed" / f"nifty_{interval}_processed.csv"
    if not processed_file.exists():
        print(f"Error: {processed_file} not found.")
        return

    df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
    series = df['Close']
    
    # Split data
    split_point = int(len(series) * 0.8)
    train, test = series[0:split_point], series[split_point:]
    
    experiment_name = f"NIFTY_ARIMA_{interval}"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"ARIMA_{interval}_Model"):
        # Log last record time and price for analysis
        last_ts = pd.to_datetime(series.index[-1])
        last_time = last_ts.strftime("%Y-%m-%d %H:%M:%S")
        last_price = float(series.iloc[-1])
        mlflow.log_param("last_record_time", last_time)
        mlflow.log_metric("last_record_price", last_price)
        
        mlflow.log_param("interval", interval)
        mlflow.log_param("p", p)
        mlflow.log_param("d", d)
        mlflow.log_param("q", q)
        
        print(f"Training ARIMA({p},{d},{q}) for {interval}...")
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        
        # Forecast test set
        forecast_result = model_fit.forecast(steps=len(test))
        mse = mean_squared_error(test, forecast_result)
        
        # Final prediction (fit on full series)
        full_model = ARIMA(series, order=(p, d, q))
        full_model_fit = full_model.fit()
        pred_next = float(full_model_fit.forecast(steps=1).iloc[0])
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("last_close", float(series.iloc[-1]))
        mlflow.log_metric("pred_next", pred_next)
        
        print(f"[{interval}] ARIMA Trained: MSE={mse:.4f}, Next Pred={pred_next:.4f}")
        return model_fit, mse, pred_next

if __name__ == "__main__":
    for interval in ["1h", "1d", "1w", "1m"]:
        train_arima_model(interval=interval)
