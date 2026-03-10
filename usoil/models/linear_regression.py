from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
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
from utils.mlflow_notify import notify_mlflow_run_summary

INTERVAL_TEST_SIZE = {
    "1h": 0.2,
    "1d": 0.25,
    "1w": 0.3,
    "1m": 0.25,
}

INTERVAL_FEATURES = {
    "1h": ["Close", "MA7", "MA21", "RSI", "Daily_Return", "STD7", "SMMA7", "Volume"],
    "1d": ["Close", "MA7", "MA21", "RSI", "STD7", "Daily_Return", "Volume"],
    "1w": ["Close", "MA21", "RSI", "STD7", "Volume"],
    "1m": ["Close", "MA21", "RSI", "STD7", "Volume"],
}

RIDGE_ALPHAS = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0]


def _resolve_features(df, interval):
    preferred = INTERVAL_FEATURES.get(interval, INTERVAL_FEATURES["1h"])
    features = [col for col in preferred if col in df.columns]
    if len(features) < 2:
        fallback = ["Close", "MA7", "MA21", "RSI", "Daily_Return", "STD7", "SMMA7", "Volume"]
        features = [col for col in fallback if col in df.columns]
    return features


def _choose_model(X_train, y_train, interval):
    candidates = [("linear", LinearRegression())]
    if interval in {"1d", "1w"}:
        candidates.extend((f"ridge_{alpha:g}", Ridge(alpha=alpha, solver="lsqr")) for alpha in RIDGE_ALPHAS)

    max_splits = min(4, len(X_train) - 1)
    if max_splits < 2:
        model_name, model = candidates[0]
        model.fit(X_train, y_train)
        return model, model_name, None

    splitter = TimeSeriesSplit(n_splits=max_splits)
    best_score = float("inf")
    best_name = "linear"
    best_estimator = LinearRegression()

    for model_name, estimator in candidates:
        fold_scores = []
        for fold_train_idx, fold_val_idx in splitter.split(X_train):
            fold_model = clone(estimator)
            fold_model.fit(X_train.iloc[fold_train_idx], y_train.iloc[fold_train_idx])
            fold_pred = fold_model.predict(X_train.iloc[fold_val_idx])
            fold_scores.append(mean_absolute_error(y_train.iloc[fold_val_idx], fold_pred))

        cv_mae = float(np.mean(fold_scores))
        if cv_mae < best_score:
            best_score = cv_mae
            best_name = model_name
            best_estimator = clone(estimator)

    best_estimator.fit(X_train, y_train)
    return best_estimator, best_name, best_score


def train_linear_regression(interval="1h", test_size=None):
    """
    Trains a Linear Regression model to predict the next period Close price for a given interval.
    """
    processed_file = PROJECT_ROOT / "usoil" / "data" / "processed" / f"usoil_{interval}_processed.csv"
    if not processed_file.exists():
        print(f"Error: {processed_file} not found.")
        return

    df = pd.read_csv(processed_file, index_col=0, parse_dates=True)

    features = _resolve_features(df, interval)
    effective_test_size = INTERVAL_TEST_SIZE.get(interval, 0.2) if test_size is None else test_size

    # Train on rows with known next-close targets, but keep the latest row for true next-step inference.
    train_df = df.copy()
    train_df['Target'] = train_df['Close'].shift(-1)
    train_df = train_df.dropna(subset=['Target'])

    X = train_df[features]
    y = train_df['Target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=effective_test_size, shuffle=False)
    
    experiment_name = f"USOIL_LR_{interval}"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"LR_{interval}_Tuned"):
        # Log last record time and price for analysis
        last_ts = pd.to_datetime(df.index[-1])
        last_time = last_ts.strftime("%Y-%m-%d %H:%M:%S")
        last_price = float(df['Close'].iloc[-1])
        mlflow.log_param("last_record_time", last_time)
        mlflow.log_metric("last_record_price", last_price)
        
        mlflow.log_param("interval", interval)
        mlflow.log_param("features", features)
        mlflow.log_param("test_size", effective_test_size)

        model, model_name, cv_mae = _choose_model(X_train, y_train, interval)
        mlflow.log_param("selected_model", model_name)
        if cv_mae is not None:
            mlflow.log_metric("cv_mae", float(cv_mae))
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        last_features = df[features].iloc[[-1]]
        pred_next = model.predict(last_features)[0]
        
        mlflow.log_param("last_close_price", f"{last_price:.8f}")
        mlflow.log_param("predicted_price", f"{float(pred_next):.8f}")

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("pred_next", float(pred_next))
        mlflow.log_metric("last_close_price", last_price)
        mlflow.log_metric("predicted_price", float(pred_next))
        
        mlflow_sklearn = importlib.import_module("mlflow.sklearn")
        mlflow_sklearn.log_model(model, "model", registered_model_name=f"USOIL_LR_{interval}")
        
        print(f"[{interval}] LR Trained: MSE={mse:.4f}, Next Pred={pred_next:.4f}")
        notify_mlflow_run_summary(asset_name="USOIL", model_name="LR", interval=interval)
        return model, mse, mae, pred_next
