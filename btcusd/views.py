from django.shortcuts import render
import pandas as pd
import pathlib
import plotly.graph_objects as go
import plotly.io as pio
import os
from datetime import datetime, timezone
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{MLFLOW_DB_PATH}"


def _get_predicted_price(run):
    """Return predicted price from newest schema first, then legacy metric."""
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


def btcusd_dashboard(request):
    """Main BTCUSD Dashboard with interval selection cards and top predictions."""
    # Get latest data for 1h to show the main graph
    processed_file = PROJECT_ROOT / "btcusd" / "data" / "processed" / "btcusd_1h_processed.csv"
    
    context = {
        'asset_name': 'BTCUSD',
        'intervals': ['1h', '1d', '1w', '1m'],
        'predictions_lr': {},
        'predictions_arima': {},
        'running_signal_label': 'N/A',
        'running_signal_class': 'secondary',
        'running_call_value': 'N/A',
        'running_call_time': 'N/A',
        'running_profit': 'N/A',
        'running_profit_class': 'secondary',
    }
    
    # Fetch predictions for each interval from MLflow
    client = MlflowClient()
    
    latest_call = None
    for interval in context['intervals']:
        exp_name = f"BTCUSD_LR_{interval}"
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
                    context['predictions_lr'][interval] = {
                        'price': f"{pred:,.2f}",
                        'change': f"{change:+,.2f}",
                        'pct': f"{pct:+.2f}%",
                        'is_positive': change >= 0
                    }
                    if interval == "1h":
                        latest_call = {
                            "side": "BUY" if pred > last_close else "SELL",
                            "trigger_price": float(last_close),
                            "trigger_time": run.data.params.get("last_record_time", "N/A"),
                        }

        exp_name = f"BTCUSD_ARIMA_{interval}"
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
                    context['predictions_arima'][interval] = {
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
        if latest_call:
            context["running_call_value"] = f"{latest_call['trigger_price']:,.2f}"
            context["running_call_time"] = latest_call["trigger_time"]
            context["running_signal_label"] = f"{latest_call['side']} (Live)"
            context["running_signal_class"] = "success" if latest_call["side"] == "BUY" else "danger"
            running_pnl = (
                latest_price - latest_call["trigger_price"]
                if latest_call["side"] == "BUY"
                else latest_call["trigger_price"] - latest_price
            )
            if running_pnl > 0:
                pnl_class = "success"
            elif running_pnl < 0:
                pnl_class = "danger"
            else:
                pnl_class = "secondary"
            context["running_profit"] = f"{running_pnl:+,.2f}"
            context["running_profit_class"] = pnl_class
        
        # Create a simple interactive graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['Close'].iloc[-100:], 
                                mode='lines', name='Price', line=dict(color='#f7931a', width=3)))
        
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
    
    return render(request, 'btcusd/dashboard.html', context)

def _interval_detail(request, interval, model_key):
    processed_file = PROJECT_ROOT / "btcusd" / "data" / "processed" / f"btcusd_{interval}_processed.csv"
    normalized_model = model_key.upper()
    selected_model = normalized_model if normalized_model in {"LR", "ARIMA"} else "LR"
    selected_model_label = "Linear Regression" if selected_model == "LR" else "ARIMA"
    
    context = {
        'interval': interval,
        'asset_name': 'BTCUSD',
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
            experiment = client.get_experiment_by_name(f"BTCUSD_{selected_model}_{interval}")
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

        # Interactive Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index[-100:],
                open=df['Open'].iloc[-100:],
                high=df['High'].iloc[-100:],
                low=df['Low'].iloc[-100:],
                close=df['Close'].iloc[-100:],
                name='Market Data'))
        
        # Add MA7
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['MA7'].iloc[-100:], 
                                mode='lines', name='MA7', line=dict(color='rgba(255,255,255,0.5)', width=1)))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        context['chart_html'] = pio.to_html(fig, full_html=False)
        
        # Real ROI & Model Comparison (LR vs ARIMA)
        try:
            client = MlflowClient()
            experiments = {
                "LR": f"BTCUSD_LR_{interval}",
                "ARIMA": f"BTCUSD_ARIMA_{interval}"
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
                        run_time = run.data.params.get('last_record_time', 'N/A')
                        
                        if last_close is not None and pred_next is not None:
                            signal = "BUY" if pred_next > last_close else "SELL"
                            temp_signals.append({
                                'time': run_time,
                                'last_close': last_close,
                                'pred_next': pred_next,
                                'signal': signal,
                            })
                    
                    # Process signals to determine wins/losses (comparing current signal with next available actual close)
                    for i in range(len(temp_signals) - 1):
                        curr = temp_signals[i+1] # Earlier run
                        next_actual = temp_signals[i]['last_close'] # Later run acts as the "actual" outcome for the earlier signal
                        
                        is_win = False
                        if curr['signal'] == "BUY":
                            is_win = next_actual > curr['last_close']
                        else:
                            is_win = next_actual <= curr['last_close']
                        
                        profit = abs(next_actual - curr['last_close']) if is_win else -abs(next_actual - curr['last_close'])
                        total_profit += profit
                        if is_win: wins += 1
                        total_resolved += 1
                        
                        signals.append({
                            'time': curr['time'],
                            'last_close': f"{curr['last_close']:,.2f}",
                            'pred_next': f"{curr['pred_next']:,.2f}",
                            'predicted': f"{curr['pred_next']:,.2f}",
                            'signal': curr['signal'],
                            'signal_class': "success" if curr['signal'] == "BUY" else "danger",
                            'result': "WIN" if is_win else "LOSS",
                            'result_class': "success" if is_win else "danger",
                            'profit': f"{profit:+,.2f}"
                        })

                    win_rate = (wins / total_resolved * 100) if total_resolved > 0 else 0
                    comparison_data[model_key] = {
                        'profit': f"{total_profit:+,.2f}",
                        'win_rate': f"{win_rate:.1f}%",
                        'signals': signals[:5] # Show last 5
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

    return render(request, 'btcusd/interval_detail.html', context)


def interval_detail(request, interval):
    return _interval_detail(request, interval, "LR")


def interval_detail_lr(request, interval):
    return _interval_detail(request, interval, "LR")


def interval_detail_arima(request, interval):
    return _interval_detail(request, interval, "ARIMA")
