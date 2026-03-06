import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Input
from sklearn.preprocessing import MinMaxScaler
import mlflow
import os
import matplotlib.pyplot as plt

# Explicitly import mlflow.tensorflow for autologging
import mlflow.tensorflow

# Set up MLflow
# Using local tracking for now to avoid server connectivity issues in sandbox
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("BTCUSD_Forecasting")
mlflow.tensorflow.autolog()

def prepare_sequences(data, target_col='Close', window_size=60):
    """
    Prepare sequences for RNN training.
    """
    scaler = MinMaxScaler()
    # Use features: Close, MA7, MA21, Daily_Return, Volume
    features = ['Close', 'MA7', 'MA21', 'Daily_Return', 'Volume']
    scaled_data = scaler.fit_transform(data[features])
    
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i, 0])  # Predicting 'Close' price
        
    return np.array(X), np.array(y), scaler

def build_model(input_shape, model_type='LSTM', units=50, dropout=0.2):
    """
    Build LSTM or GRU model using modern Keras Input object.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    if model_type == 'LSTM':
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units, return_sequences=False))
        model.add(Dropout(dropout))
    else:
        model.add(GRU(units=units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(GRU(units=units, return_sequences=False))
        model.add(Dropout(dropout))
        
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model_type='LSTM', window_size=60, epochs=1, batch_size=64):
    # Load processed data
    data_path = "data/processed/btcusd_processed.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run ingestion.py first.")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # Take only the last 500 rows for faster training in sandbox verification
    df = df.tail(500)
    
    X, y, scaler = prepare_sequences(df, window_size=window_size)
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # MLflow tracking
    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        
        model = build_model((X_train.shape[1], X_train.shape[2]), model_type=model_type)
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate
        test_loss = model.evaluate(X_test, y_test)
        mlflow.log_metric("test_mse", test_loss)
        
        # Log model with registry
        model_info = mlflow.tensorflow.log_model(
            model, 
            "model",
            registered_model_name="BTCUSD_RNN_Model"
        )
        
        print(f"Training complete. Test Loss (MSE): {test_loss}")
        print(f"Model registered as 'BTCUSD_RNN_Model' at: {model_info.model_uri}")
        return model, history

if __name__ == "__main__":
    # Ensure MLflow server is running or use local tracking
    # For this exercise, we'll use default local tracking if no server
    try:
        train_model(model_type='LSTM', epochs=5)
    except Exception as e:
        print(f"Training failed: {e}")
        print("Note: Ensure MLflow tracking server is reachable if set.")
