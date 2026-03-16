from django.shortcuts import render
import pandas as pd
import pathlib
import plotly.graph_objects as go
import plotly.io as pio
import os
import mlflow
from datetime import datetime, timezone
from mlflow.tracking import MlflowClient

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{MLFLOW_DB_PATH}"


def _get_predicted_price(run):
    raw = (
        run.data.params.get("predicted_price")
        or run.data.metrics.get("predicted_price")
        or run.data.metrics.get("pred_next")
    )
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _get_last_close_price(run):
    raw = (
        run.data.params.get("last_close_price")
        or run.data.metrics.get("last_close_price")
        or run.data.metrics.get("last_record_price")
        or run.data.metrics.get("last_close")
    )
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _fmt_run_timestamp(ms):
    if not ms:
        return "N/A"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def nifty_dashboard(request):
    """Main Nifty Dashboard with interval selection cards and top predictions."""
    # Get latest data for 1h to show the main graph
    processed_file = PROJECT_ROOT / "nifty" / "data" / "processed" / "nifty_1h_processed.csv"
    
    context = {
        'asset_name': 'Nifty 50',
        'intervals': ['1h', '1d', '1w', '1m'],
        'predictions': {},
    }
    
    # Fetch predictions for each interval from MLflow
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    
    for interval in context['intervals']:
        exp_name = f"NIFTY_LR_{interval}"
        experiment = client.get_experiment_by_name(exp_name)
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=1
            )
            if runs:
                run = runs[0]
                pred = _get_predicted_price(run)
                last_close = _get_last_close_price(run)
                if pred and last_close:
                    change = pred - last_close
                    pct = (change / last_close) * 100
                    context['predictions'][interval] = {
                        'price': f"{pred:,.2f}",
                        'change': f"{change:+,.2f}",
                        'pct': f"{pct:+.2f}%",
                        'is_positive': change >= 0
                    }

    if processed_file.exists():
        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = latest_price - prev_price
        pct_change = (price_change / prev_price) * 100
        
        context.update({
            'latest_price': f"{latest_price:,.2f}",
            'price_change': f"{price_change:+,.2f}",
            'pct_change': f"{pct_change:+.2f}%",
            'is_positive': price_change >= 0,
        })
        
        # Create a simple interactive graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['Close'].iloc[-100:], 
                                mode='lines', name='Price', line=dict(color='#007bff', width=3))) # India Blue
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            height=400,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#333'),
        )
        
        context['graph_html'] = pio.to_html(fig, full_html=False)
    
    return render(request, 'nifty/dashboard.html', context)

def _interval_detail(request, interval, model_key):
    processed_file = PROJECT_ROOT / "nifty" / "data" / "processed" / f"nifty_{interval}_processed.csv"
    normalized_model = model_key.upper()
    selected_model = normalized_model if normalized_model in {"LR", "ARIMA"} else "LR"
    selected_model_label = "Linear Regression" if selected_model == "LR" else "ARIMA"
    
    context = {
        'interval': interval,
        'asset_name': 'Nifty 50',
        'forecast_price': "N/A",
        'last_run_time': 'N/A',
        'selected_model': selected_model,
        'selected_model_label': selected_model_label,
        'completed_runs': [],
    }
    
    if processed_file.exists():
        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
        
        # Latest data
        latest_price = df['Close'].iloc[-1]
        context['latest_price'] = f"{latest_price:,.2f}"
        
        # Get Forecast from MLflow
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name(f"NIFTY_{selected_model}_{interval}")
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["attributes.start_time DESC"],
                    max_results=20
                )
                if runs:
                    latest_run = runs[0]
                    pred_value = _get_predicted_price(latest_run)
                    if pred_value is not None:
                        context['forecast_price'] = f"{pred_value:,.2f}"
                    context['last_run_time'] = latest_run.data.params.get('last_record_time', 'N/A')

                    for run in runs:
                        if (run.info.status or "").upper() != "FINISHED":
                            continue

                        run_pred = _get_predicted_price(run)
                        run_last_close = _get_last_close_price(run)
                        metrics = run.data.metrics
                        mae_value = metrics.get("mae")
                        mse_value = metrics.get("mse")
                        context["completed_runs"].append({
                            "run_id": run.info.run_id,
                            "start_time": _fmt_run_timestamp(run.info.start_time),
                            "end_time": _fmt_run_timestamp(run.info.end_time),
                            "status": run.info.status,
                            "last_record_time": run.data.params.get("last_record_time", "N/A"),
                            "last_close_price": f"{run_last_close:,.2f}" if run_last_close is not None else "N/A",
                            "predicted_price": f"{run_pred:,.2f}" if run_pred is not None else "N/A",
                            "mse": f"{mse_value:,.2f}" if mse_value is not None else "N/A",
                            "mae": f"{mae_value:,.2f}" if mae_value is not None else "N/A",
                        })
        except Exception as e:
            print(f"Error fetching forecast: {e}")

        # Real ROI & Model Comparison (LR vs ARIMA)
        try:
            client = MlflowClient()
            experiments = {
                "LR": f"NIFTY_LR_{interval}",
                "ARIMA": f"NIFTY_ARIMA_{interval}"
            }
            
            comparison_data = {}
            for model_key, exp_name in experiments.items():
                exp = client.get_experiment_by_name(exp_name)
                if exp:
                    runs = client.search_runs(
                        experiment_ids=[exp.experiment_id],
                        order_by=["attributes.start_time DESC"],
                        max_results=20
                    )
                    
                    signals = []
                    wins = 0
                    total_resolved = 0
                    total_profit = 0.0
                    
                    # Convert runs to signals for processing
                    temp_signals = []
                    for run in runs:
                        last_close = _get_last_close_price(run)
                        pred_next = _get_predicted_price(run)
                        run_time = _fmt_run_timestamp(run.info.start_time)
                        metrics = run.data.metrics
                        
                        if last_close is not None and pred_next is not None:
                            signal = "BUY" if pred_next > last_close else "SELL"
                            temp_signals.append({
                                'time': run_time,
                                'last_close': last_close,
                                'pred_next': pred_next,
                                'signal': signal,
                                'mae': metrics.get("mae"),
                                'mse': metrics.get("mse"),
                            })
                    
                    # Process signals to determine wins/losses
                    changeover_signals = []
                    for idx, curr in enumerate(temp_signals):
                        baseline_close = curr["last_close"]
                        
                        signal_row = {
                            "time": curr["time"],
                            "last_close": f"{baseline_close:,.2f}",
                            "pred_next": f"{curr['pred_next']:,.2f}",
                            "predicted": f"{curr['pred_next']:,.2f}",
                            "signal": curr["signal"],
                            "signal_class": "success" if curr["signal"] == "BUY" else "danger",
                            "mse": f"{curr['mse']:,.2f}" if curr['mse'] is not None else "N/A",
                            "mae": f"{curr['mae']:,.2f}" if curr['mae'] is not None else "N/A",
                        }

                        if idx == 0:
                            signal_row.update({
                                "actual_next": "N/A",
                                "result": "PENDING",
                                "result_class": "warning",
                                "profit": "N/A",
                            })
                            signals.append(signal_row)
                            changeover_signals.append(signal_row)
                            continue

                        next_actual = temp_signals[idx - 1]["last_close"]
                        
                        is_win = False
                        if curr["signal"] == "BUY":
                            is_win = next_actual > curr["last_close"]
                        else:
                            is_win = next_actual <= curr["last_close"]
                        
                        profit = abs(next_actual - curr["last_close"]) if is_win else -abs(next_actual - curr["last_close"])
                        total_profit += profit
                        if is_win: wins += 1
                        total_resolved += 1
                        
                        signal_row.update({
                            "actual_next": f"{next_actual:,.2f}",
                            "result": "WIN" if is_win else "LOSS",
                            "result_class": "success" if is_win else "danger",
                            "profit": f"{profit:+,.2f}",
                        })
                        signals.append(signal_row)

                        # Check for signal flip
                        if idx + 1 < len(temp_signals) and curr["signal"] != temp_signals[idx + 1]["signal"]:
                            changeover_signals.append(signal_row)

                    win_rate = (wins / total_resolved * 100) if total_resolved > 0 else 0
                    comparison_data[model_key] = {
                        'profit': f"{total_profit:+,.2f}",
                        'win_rate': f"{win_rate:.1f}%",
                        'signals': signals[:20], # Show last 20
                        'changeover_signals': changeover_signals[:10] # Show last 10 changes
                    }
            
            if selected_model in comparison_data:
                context['comparison'] = {selected_model: comparison_data[selected_model]}
            else:
                context['comparison'] = {}
            
            # Update summary stats with real data if available
            if selected_model in comparison_data:
                context['roi_estimate'] = f"{comparison_data[selected_model]['win_rate']} Win Rate"
                context['ab_test_result'] = f"{selected_model} ({comparison_data[selected_model]['profit']} pts)"

        except Exception as e:
            print(f"Error fetching ROI data: {e}")
            context['comparison'] = {}

        # Interactive Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index[-100:],
                open=df['Open'].iloc[-100:],
                high=df['High'].iloc[-100:],
                low=df['Low'].iloc[-100:],
                close=df['Close'].iloc[-100:],
                name='Market Data'))
        
        # Add Technical Indicators
        if 'SMA_20' in df.columns:
            fig.add_trace(go.Scatter(x=df.index[-100:], y=df['SMA_20'].iloc[-100:], 
                                    mode='lines', name='SMA 20', line=dict(color='rgba(0,123,255,0.5)', width=1))) # India Blue translucent
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=30, b=0),
            title=f"Nifty 50 {interval} Technical Analysis"
        )
        context['chart_html'] = pio.to_html(fig, full_html=False)
        
        # Additional context for cards
        context.update({
            'drift_score': 'Low (0.08)',
        })
    
    return render(request, 'nifty/interval_detail.html', context)


def interval_detail(request, interval):
    return _interval_detail(request, interval, "LR")


def interval_detail_lr(request, interval):
    return _interval_detail(request, interval, "LR")


def interval_detail_arima(request, interval):
    return _interval_detail(request, interval, "ARIMA")
