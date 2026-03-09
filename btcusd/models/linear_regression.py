from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import mlflow
import importlib
import os
import pathlib

# Path setup
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
# Central MLflow DB
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{MLFLOW_DB_PATH}"

def train_linear_regression(interval="1h", test_size=0.2):
    """
    Trains a Linear Regression model to predict the next period Close price for a given interval.
    """
    processed_file = PROJECT_ROOT / "btcusd" / "data" / "processed" / f"btcusd_{interval}_processed.csv"
    if not processed_file.exists():
        print(f"Error: {processed_file} not found.")
        return

    df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
    
    # Feature Engineering
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    
    features = ['Close', 'MA7', 'MA21', 'Daily_Return', 'Volume', 'STD7', 'SMMA7', 'RSI']
    X = df[features]
    y = df['Target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    experiment_name = f"BTCUSD_LR_{interval}"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"LR_{interval}_Baseline"):
        # Log last record time and price for analysis
        last_ts = pd.to_datetime(df.index[-1])
        last_time = last_ts.strftime("%Y-%m-%d %H:%M:%S")
        last_price = float(df['Close'].iloc[-1])
        mlflow.log_param("last_record_time", last_time)
        mlflow.log_metric("last_record_price", last_price)
        
        mlflow.log_param("interval", interval)
        mlflow.log_param("features", features)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        last_features = X.iloc[-1].values.reshape(1, -1)
        pred_next = model.predict(last_features)[0]
        
        mlflow.log_param("last_close_price", f"{last_price:.8f}")
        mlflow.log_param("predicted_price", f"{float(pred_next):.8f}")

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("pred_next", float(pred_next))
        mlflow.log_metric("last_close_price", last_price)
        mlflow.log_metric("predicted_price", float(pred_next))
        
        mlflow_sklearn = importlib.import_module("mlflow.sklearn")
        mlflow_sklearn.log_model(model, "model", registered_model_name=f"BTCUSD_LR_{interval}")
        
        print(f"[{interval}] LR Trained: MSE={mse:.4f}, Next Pred={pred_next:.4f}")
        return model, mse, mae, pred_next
