from django.shortcuts import render
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

def dashboard_overview(request):
    """
    Fetches experiment data and model registry info from MLflow for the dashboard.
    """
    # Initialize MLflow Client
    client = MlflowClient()
    
    # Get Experiment Info
    experiment_name = "BTCUSD_Forecasting"
    experiment = client.get_experiment_by_name(experiment_name)
    
    latest_prediction = "N/A"
    model_accuracy = "N/A"
    runs_data = []

    if experiment:
        # Search for runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_mse ASC"],
            max_results=5
        )
        
        for run in runs:
            runs_data.append({
                'run_id': run.info.run_id,
                'status': run.info.status,
                'model_type': run.data.params.get('model_type', 'N/A'),
                'mse': run.data.metrics.get('test_mse', 'N/A'),
                'start_time': pd.to_datetime(run.info.start_time, unit='ms').strftime('%Y-%m-%d %H:%M')
            })
        
        if runs:
            # Get best run for display
            best_run = runs[0]
            model_accuracy = f"{100 * (1 - best_run.data.metrics.get('test_mse', 0)):.2f}%" if 'test_mse' in best_run.data.metrics else "N/A"

    # Get Registered Model Info
    try:
        registered_model = client.get_registered_model("BTCUSD_RNN_Model")
        latest_versions = registered_model.latest_versions
    except Exception:
        latest_versions = []

    context = {
        'latest_prediction': latest_prediction,
        'model_accuracy': model_accuracy,
        'runs': runs_data,
        'latest_versions': latest_versions
    }
    
    return render(request, 'dashboard/overview.html', context)
