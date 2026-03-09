import os
import subprocess
from django.shortcuts import render, redirect
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from django.conf import settings
import json
import plotly.graph_objects as go
from plotly.offline import plot

def dashboard_overview(request):
    """
    Fetches experiment data and model registry info from MLflow for the dashboard.
    """
    # Configure MLflow to use the central sqlite DB in the project root
    project_root = settings.BASE_DIR.parent
    mlflow_db_path = os.path.join(project_root, "mlflow.db")
    tracking_uri = f"sqlite:///{mlflow_db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Initialize MLflow Client
    client = MlflowClient()
    
    latest_prediction_lr = "N/A"
    latest_prediction_arima = "N/A"
    latest_price = "N/A"
    model_accuracy = "N/A"
    runs_data = []

    # Get Experiments for BTCUSD across common intervals
    experiments = {
        "LR": "BTCUSD_LR_1h",
        "ARIMA": "BTCUSD_ARIMA_1h"
    }

    for model_key, exp_name in experiments.items():
        exp = client.get_experiment_by_name(exp_name)
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=10
            )
            for run in runs:
                # In the new structure, metrics are 'mse' and 'pred_next'
                mse = run.data.metrics.get('mse', 'N/A')
                pred = run.data.metrics.get('pred_next', 'N/A')
                
                if model_key == "LR" and latest_prediction_lr == "N/A" and pred != "N/A":
                    latest_prediction_lr = f"{pred:.2f}"
                elif model_key == "ARIMA" and latest_prediction_arima == "N/A" and pred != "N/A":
                    latest_prediction_arima = f"{pred:.2f}"

                runs_data.append({
                    'run_id': run.info.run_id,
                    'status': run.info.status,
                    'model_type': model_key,
                    'mse': mse,
                    'start_time': pd.to_datetime(run.info.start_time, unit='ms').strftime('%Y-%m-%d %H:%M')
                })

    # Sort all runs by start time
    runs_data = sorted(runs_data, key=lambda x: x['start_time'], reverse=True)

    # Visualization and Latest Price
    plot_div = ""
    try:
        # Update path to use the new btcusd app data directory
        data_path = os.path.join(project_root, "btcusd", "data", "processed", "btcusd_1h_processed.csv")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            if not df.empty:
                latest_price = f"{df.iloc[-1]['Close']:.2f}"
                
                # Create Plotly Chart
                fig = go.Figure()
                # Last 48 hours for better visibility
                df_recent = df.tail(48)
                fig.add_trace(go.Scatter(
                    x=df_recent.index, 
                    y=df_recent['Close'], 
                    mode='lines+markers', 
                    name='Actual Price',
                    line=dict(color='#f7931a', width=3),  # BTC Orange
                    marker=dict(size=6, color='#f7931a')
                ))
                
                # Add Prediction Points
                last_time = df_recent.index[-1]
                next_time = last_time + pd.Timedelta(hours=1)
                
                if latest_prediction_lr != "N/A":
                    fig.add_trace(go.Scatter(x=[next_time], y=[float(latest_prediction_lr)], 
                                           mode='markers', name='LR Prediction', marker=dict(size=14, symbol='star', color='#00ff00')))
                if latest_prediction_arima != "N/A":
                    fig.add_trace(go.Scatter(x=[next_time], y=[float(latest_prediction_arima)], 
                                           mode='markers', name='ARIMA Prediction', marker=dict(size=12, symbol='diamond', color='#00bcff')))
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    title=dict(text="BTCUSD Price & Next Hour Forecast", font=dict(size=20)),
                    xaxis=dict(
                        title="Time",
                        gridcolor='#333',
                        zerolinecolor='#333'
                    ),
                    yaxis=dict(
                        title="Price (USD)",
                        gridcolor='#333',
                        zerolinecolor='#333'
                    ),
                    margin=dict(l=20, r=20, t=60, b=20),
                    height=500,
                    legend=dict(
                        bgcolor='rgba(0,0,0,0.5)',
                        bordercolor='#444'
                    )
                )
                plot_div = plot(fig, output_type='div', include_plotlyjs=True)

    except Exception as e:
        print(f"Error reading data or creating plot: {e}")

    # Best Model Accuracy (Lowest MSE)
    valid_mse_runs = [r for r in runs_data if isinstance(r['mse'], (int, float))]
    if valid_mse_runs:
        best_run = min(valid_mse_runs, key=lambda x: x['mse'])
        model_accuracy = f"{best_run['mse']:.2f} (MSE - {best_run['model_type']})"

    # Get Registered Model Info
    registered_models = ["BTCUSD_Linear_Regression", "BTCUSD_ARIMA"]
    latest_versions = []
    for model_name in registered_models:
        try:
            rm = client.get_registered_model(model_name)
            versions = rm.latest_versions
            if versions:
                latest_versions.extend(versions)
        except Exception:
            continue

    context = {
        'latest_price': latest_price,
        'latest_prediction_lr': latest_prediction_lr,
        'latest_prediction_arima': latest_prediction_arima,
        'model_accuracy': model_accuracy,
        'runs': runs_data[:10],
        'latest_versions': latest_versions,
        'plot_div': plot_div
    }
    
    return render(request, 'dashboard/overview.html', context)

def run_pipeline_view(request):
    """
    Triggers the ML pipeline from the dashboard.
    """
    if request.method == "POST":
        try:
            pipeline_path = os.path.join(settings.BASE_DIR.parent, "ml", "pipeline.py")
            # Run in background
            subprocess.Popen(["python3", pipeline_path, "--models", "lr,arima"])
        except Exception as e:
            print(f"Error triggering pipeline: {e}")
            
    return redirect('dashboard_overview')

def drift_monitoring(request):
    """
    Drift Monitoring View - Analyzes price distributions and error trends.
    """
    ml_dir = os.path.join(settings.BASE_DIR.parent, "ml")
    data_path = os.path.join(ml_dir, "data", "processed", "btcusd_processed.csv")
    
    drift_status = "Unknown"
    drift_details = []
    plot_div = ""

    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            if not df.empty and len(df) > 48:
                # 1. Price Drift Check (Comparing last 24h vs previous 24h distribution)
                recent_24h = df['Close'].tail(24)
                prev_24h = df['Close'].iloc[-48:-24]
                
                price_change_pct = abs((recent_24h.mean() - prev_24h.mean()) / prev_24h.mean() * 100)
                
                # Simple drift detection logic
                if price_change_pct > 10:
                    drift_status = "Critical Drift"
                elif price_change_pct > 5:
                    drift_status = "Warning: Minor Drift"
                else:
                    drift_status = "Stable"
                
                drift_details.append({
                    'feature': 'Close Price',
                    'recent_mean': f"${recent_24h.mean():.2f}",
                    'prev_mean': f"${prev_24h.mean():.2f}",
                    'change': f"{price_change_pct:.2f}%",
                    'status': "Alert" if price_change_pct > 5 else "OK"
                })

                # Create Drift Visualization
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=prev_24h, name='Previous 24h', opacity=0.75, marker_color='#636EFA'))
                fig.add_trace(go.Histogram(x=recent_24h, name='Recent 24h', opacity=0.75, marker_color='#EF553B'))
                
                fig.update_layout(
                    title="Price Distribution Comparison (Drift Detection)",
                    xaxis_title="Price (USD)",
                    yaxis_title="Frequency",
                    barmode='overlay',
                    template="plotly_white",
                    height=450
                )
                plot_div = plot(fig, output_type='div', include_plotlyjs=True)
                
        except Exception as e:
            print(f"Error in drift calculation: {e}")

    context = {
        'drift_status': drift_status,
        'drift_details': drift_details,
        'plot_div': plot_div,
        'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
    }
    return render(request, 'dashboard/drift.html', context)
