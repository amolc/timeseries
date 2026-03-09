import os
from django.shortcuts import render
from .models import ROIMetric
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from django.conf import settings

def roi_index(request):
    """
    Displays simulated ROI and financial impact of the forecasting model.
    """
    # Configure MLflow to use the central sqlite DB in the project root
    project_root = settings.BASE_DIR.parent
    mlflow_db_path = os.path.join(project_root, "mlflow.db")
    tracking_uri = f"sqlite:///{mlflow_db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Initialize MLflow Client
    client = MlflowClient()
    
    # Get Experiments for BTCUSD across common intervals
    experiments = {
        "LR": "BTCUSD_LR_1h",
        "ARIMA": "BTCUSD_ARIMA_1h"
    }

    lr_signals = []
    arima_signals = []

    lr_profit = 0.0
    arima_profit = 0.0

    for model_key, exp_name in experiments.items():
        exp = client.get_experiment_by_name(exp_name)
        if exp:
            # Search for runs (order by start time to get most recent first)
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=50
            )
            
            seen_minutes = set()
            temp_signals = []
            for run in runs:
                # In the new structure, metrics are 'last_record_price' and 'pred_next'
                last_close = run.data.metrics.get('last_record_price')
                pred_next = run.data.metrics.get('pred_next')
                
                dt = pd.to_datetime(run.info.start_time, unit='ms')
                time_key = dt.strftime('%Y-%m-%d %H:%M')
                
                if time_key in seen_minutes:
                    continue
                
                if last_close is not None and pred_next is not None:
                    seen_minutes.add(time_key)
                    signal = "BUY" if last_close < pred_next else "SELL"
                    signal_class = "success" if signal == "BUY" else "danger"
                    
                    # Potential Profit (Points)
                    potential_profit = abs(pred_next - last_close)
                    
                    temp_signals.append({
                        'time': time_key,
                        'last_close': last_close,
                        'pred_next': pred_next,
                        'signal': signal,
                        'signal_class': signal_class,
                        'potential_profit': potential_profit,
                        'raw_time': run.info.start_time
                    })

            processed_signals = []
            wins = 0
            total_resolved = 0
            model_profit = 0.0
            
            for i in range(len(temp_signals)):
                curr = temp_signals[i]
                result = "PENDING"
                result_class = "secondary"
                
                if i > 0:
                    actual_later = temp_signals[i-1]['last_close']
                    is_win = False
                    if curr['signal'] == "BUY":
                        is_win = actual_later > curr['last_close']
                    else:
                        is_win = actual_later <= curr['last_close']
                    
                    result = "WIN" if is_win else "LOSS"
                    result_class = "success" if is_win else "danger"
                    
                    if is_win:
                        wins += 1
                        # If win, we add the actual profit points (actual movement)
                        model_profit += abs(actual_later - curr['last_close'])
                    
                    total_resolved += 1
                
                processed_signals.append({
                    'time': curr['time'],
                    'last_close': f"{curr['last_close']:.2f}",
                    'pred_next': f"{curr['pred_next']:.2f}",
                    'signal': curr['signal'],
                    'signal_class': curr['signal_class'],
                    'potential_profit': f"{curr['potential_profit']:.2f}",
                    'result': result,
                    'result_class': result_class
                })

            win_rate = (wins / total_resolved * 100) if total_resolved > 0 else 0
            
            if model_key == "LR":
                lr_signals = processed_signals[:20]
                lr_win_rate = f"{win_rate:.1f}%"
                lr_profit = model_profit
            else:
                arima_signals = processed_signals[:20]
                arima_win_rate = f"{win_rate:.1f}%"
                arima_profit = model_profit

    latest_roi = ROIMetric.objects.order_by('-calculated_at').first()
    
    context = {
        'latest_roi': latest_roi,
        'history': ROIMetric.objects.all().order_by('-calculated_at')[:10],
        'lr_signals': lr_signals,
        'arima_signals': arima_signals,
        'lr_win_rate': lr_win_rate if 'lr_win_rate' in locals() else "0.0%",
        'arima_win_rate': arima_win_rate if 'arima_win_rate' in locals() else "0.0%",
        'lr_profit': f"{lr_profit:.2f}",
        'arima_profit': f"{arima_profit:.2f}"
    }
    return render(request, 'dashboard/roi_index.html', context)
