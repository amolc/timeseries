from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import pathlib
import plotly.graph_objects as go
import plotly.io as pio
import os
import mlflow
import json
from datetime import datetime, timezone
from mlflow.tracking import MlflowClient
from utils.live_price import get_last_price_payload

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


def _get_interval_predictions(client, asset, model_prefix, intervals, latest_price=None):
    predictions = {}
    for interval in intervals:
        exp_name = f"{asset}_{model_prefix}_{interval}"
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
                    # If we have a more recent price, use it for change calculation
                    baseline = latest_price if latest_price else last_close
                    change = pred - baseline
                    pct = (change / baseline) * 100
                    predictions[interval] = {
                        'price': f"{pred:,.2f}",
                        'change': f"{change:+,.2f}",
                        'pct': f"{pct:+.2f}%",
                        'is_positive': change >= 0
                    }
    return predictions


def spx500_dashboard(request):
    """Main SPX500 Dashboard with ARIMA-only targets."""
    processed_file = PROJECT_ROOT / "spx500" / "data" / "processed" / "spx500_1h_processed.csv"
    
    context = {
        'asset_name': 'SPX500',
        'intervals': ['1h', '1d', '1w', '1m'],
        'latest_price': "0.00",
        'latest_price_as_of': "N/A",
        'price_change': "+0.00",
        'pct_change': "+0.00%",
        'is_positive': True,
    }
    
    # Fetch latest price from utility
    latest_price = 0.0
    latest_price_payload = get_last_price_payload("spx500_dashboard", "SPX500", processed_file)
    if latest_price_payload.get("ok"):
        context['latest_price'] = latest_price_payload.get("latest_price", context["latest_price"])
        context['latest_price_as_of'] = latest_price_payload.get("as_of", context["latest_price_as_of"])
        latest_price = float(context['latest_price'].replace(',', ''))

    # Fetch predictions for each interval from MLflow
    client = MlflowClient()
    context['predictions_arima'] = _get_interval_predictions(
        client, "SPX500", "ARIMA", context['intervals'], latest_price=latest_price
    )

    # Signal logic from 1h ARIMA (similar to BTC/NIFTY/GOLD)
    try:
        exp_1h = client.get_experiment_by_name("SPX500_ARIMA_1h")
        if exp_1h:
            runs = client.search_runs(experiment_ids=[exp_1h.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
            if runs:
                latest_run = runs[0]
                pred_1h = _get_predicted_price(latest_run)
                last_close_1h = _get_last_close_price(latest_run)
                
                if pred_1h and last_close_1h:
                    is_buy = pred_1h > last_close_1h
                    context['running_signal_label'] = "BUY" if is_buy else "SELL"
                    context['running_signal_class'] = "success" if is_buy else "danger"
                    context['running_call_value'] = f"{pred_1h:,.2f}"
                    context['running_call_time'] = _fmt_run_timestamp(latest_run.info.start_time)
                    
                    if latest_price > 0:
                        profit = (latest_price - last_close_1h) if is_buy else (last_close_1h - latest_price)
                        context['running_profit'] = f"{profit:+,.2f}"
                        context['running_profit_class'] = "success" if profit >= 0 else "danger"
    except Exception as e:
        print(f"Error fetching 1h signal: {e}")

    if processed_file.exists():
        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
        # Fallback if utility failed
        if context['latest_price'] == "0.00":
            latest_price = df['Close'].iloc[-1]
            context['latest_price'] = f"{latest_price:,.2f}"
            
        prev_price = df['Close'].iloc[-2]
        price_change = latest_price - prev_price
        pct_change = (price_change / prev_price) * 100
        
        context.update({
            'price_change': f"{price_change:+,.2f}",
            'pct_change': f"{pct_change:+.2f}%",
            'is_positive': price_change >= 0,
        })
        
        # Graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['Close'].iloc[-100:], 
                                mode='lines', name='Price', line=dict(color='#3b82f6', width=3)))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0), height=400,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#333'),
        )
        context['graph_html'] = pio.to_html(fig, full_html=False)

    # SEO Metadata
    current_url = request.build_absolute_uri()
    context['seo'] = {
        'canonical': current_url,
        'json_ld': json.dumps({
            "@context": "https://schema.org",
            "@type": "FinancialProduct",
            "name": f"SPX500 ARIMA Intelligence",
            "description": f"Real-time ARIMA forecasting and technical analysis for SPX500.",
            "url": current_url,
            "brand": {"@type": "Brand", "name": "Quantbots Intelligence"}
        })
    }
    
    return render(request, 'spx500/dashboard.html', context)


def last_price_api(request):
    processed_file = PROJECT_ROOT / "spx500" / "data" / "processed" / "spx500_1h_processed.csv"
    response = get_last_price_payload("spx500_api", "SPX500", processed_file)
    status = 200 if response.get("ok") else 404
    return JsonResponse(response, status=status)


def _interval_detail(request, interval, model_key="ARIMA"):
    processed_file = PROJECT_ROOT / "spx500" / "data" / "processed" / f"spx500_{interval}_processed.csv"
    normalized_model = model_key.upper()
    selected_model = "ARIMA"
    selected_model_label = "ARIMA"
    
    context = {
        'interval': interval,
        'asset_name': 'SPX500',
        'forecast_price': "N/A",
        'last_run_time': 'N/A',
        'selected_model': selected_model,
        'selected_model_label': selected_model_label,
    }
    
    if processed_file.exists():
        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
        
        # Latest data
        latest_price = df['Close'].iloc[-1]
        context['latest_price'] = f"{latest_price:,.2f}"
        
        # Get Forecast from MLflow
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name(f"SPX500_{selected_model}_{interval}")
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["attributes.start_time DESC"],
                    max_results=200
                )
                if runs:
                    latest_run = runs[0]
                    pred_value = _get_predicted_price(latest_run)
                    if pred_value is not None:
                        context['forecast_price'] = f"{pred_value:,.2f}"
                    context['last_run_time'] = latest_run.data.params.get('last_record_time', 'N/A')
        except Exception as e:
            print(f"Error fetching forecast: {e}")

        # Real ROI & Model (ARIMA Only)
        try:
            client = MlflowClient()
            experiments = {
                "ARIMA": f"SPX500_ARIMA_{interval}"
            }
            
            comparison_data = {}
            for model_key, exp_name in experiments.items():
                exp = client.get_experiment_by_name(exp_name)
                if exp:
                    runs = client.search_runs(
                        experiment_ids=[exp.experiment_id],
                        order_by=["attributes.start_time DESC"],
                        max_results=200
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
                        'signals': signals[:20],
                        'changeover_signals': changeover_signals[:10]
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
                                    mode='lines', name='SMA 20', line=dict(color='rgba(0,212,255,0.5)', width=1)))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=30, b=0),
            title=f"SPX500 {interval} Technical Analysis"
        )
        context['chart_html'] = pio.to_html(fig, full_html=False)
        
        # Additional context for cards
        context.update({
            'drift_score': 'Low (0.12)',
        })
    
    return render(request, 'spx500/interval_detail.html', context)


def interval_detail(request, interval):
    return _interval_detail(request, interval, "ARIMA")


def interval_detail_arima(request, interval):
    return _interval_detail(request, interval, "ARIMA")
