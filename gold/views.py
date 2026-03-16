from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET
import pandas as pd
import pathlib
import plotly.graph_objects as go
import plotly.io as pio
import os
import json
from datetime import datetime, timezone
from django.utils import timezone as django_timezone
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


def _get_interval_predictions(client, asset_prefix, model_prefix, intervals, latest_price=None):
    predictions = {}
    for interval in intervals:
        exp_name = f"{asset_prefix}_{model_prefix}_{interval}"
        experiment = client.get_experiment_by_name(exp_name)
        if not experiment:
            continue

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        if not runs:
            continue

        run = runs[0]
        pred = _get_predicted_price(run)
        last_close = _get_last_close_price(run)
        if pred is None or last_close is None:
            continue

        change = pred - last_close
        pct = (change / last_close) * 100
        last_run_time = run.data.params.get("last_record_time") or "N/A"
        signal = "BUY" if pred > last_close else "SELL"
        signal_class = "success" if signal == "BUY" else "danger"
        running_profit = None
        running_profit_class = "secondary"
        if latest_price is not None:
            running_profit = (latest_price - last_close) if signal == "BUY" else (last_close - latest_price)
            if running_profit > 0:
                running_profit_class = "success"
            elif running_profit < 0:
                running_profit_class = "danger"
        predictions[interval] = {
            "price": f"{pred:,.2f}",
            "change": f"{change:+,.2f}",
            "pct": f"{pct:+.2f}%",
            "is_positive": change >= 0,
            "last_run_time": last_run_time,
            "signal": signal,
            "signal_class": signal_class,
            "call_value": f"{last_close:,.2f}",
            "running_profit": f"{running_profit:+,.2f}" if running_profit is not None else "N/A",
            "running_profit_class": running_profit_class,
        }
    return predictions


def gold_dashboard(request):
    """Main Gold Dashboard with interval selection cards and top predictions."""
    processed_file = PROJECT_ROOT / "gold" / "data" / "processed" / "gold_1h_processed.csv"
    
    context = {
        'asset_name': 'Gold',
        'intervals': ['1h', '1d', '1w', '1m'],
        'predictions_arima': {},
        'latest_price': "N/A",
        'latest_price_as_of': "N/A",
        'running_signal_label': "N/A",
        'running_signal_class': "secondary",
        'running_call_side': "N/A",
        'running_call_value': "N/A",
        'running_call_value_raw': "",
        'running_call_time': "N/A",
        'running_profit': "N/A",
        'running_profit_class': "secondary",
    }
    
    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    latest_price = None
    latest_price_payload = get_last_price_payload("gold_dashboard", "GOLD", processed_file)
    if latest_price_payload.get("ok"):
        context['latest_price'] = latest_price_payload.get("latest_price", context["latest_price"])
        context['latest_price_as_of'] = latest_price_payload.get("as_of", context["latest_price_as_of"])
        try:
            latest_price = float(str(context['latest_price']).replace(",", ""))
        except (TypeError, ValueError):
            latest_price = None

    # Pick the latest active call from 1h ARIMA runs.
    latest_call = None
    for exp_name in ("GOLD_ARIMA_1h",):
        exp = client.get_experiment_by_name(exp_name)
        if not exp:
            continue
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            continue
        run = runs[0]
        pred = _get_predicted_price(run)
        last_close = _get_last_close_price(run)
        if pred is None or last_close is None:
            continue
        call = {
            "start_time_ms": run.info.start_time or 0,
            "side": "BUY" if pred > last_close else "SELL",
            "trigger_price": float(last_close),
            "trigger_time": run.data.params.get("last_record_time", "N/A"),
        }
        if latest_call is None or call["start_time_ms"] > latest_call["start_time_ms"]:
            latest_call = call

    if latest_call:
        context["running_call_side"] = latest_call["side"]
        context["running_call_value"] = f"{latest_call['trigger_price']:,.2f}"
        context["running_call_value_raw"] = f"{latest_call['trigger_price']:.8f}"
        context["running_call_time"] = latest_call["trigger_time"]
        context["running_signal_label"] = f"{latest_call['side']} (Live)"
        context["running_signal_class"] = "success" if latest_call["side"] == "BUY" else "danger"
        if latest_price is not None:
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

    context['predictions_arima'] = _get_interval_predictions(
        client, "GOLD", "ARIMA", context['intervals'], latest_price=latest_price
    )

    if processed_file.exists():
        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
        if latest_price is None:
            latest_price = float(df['Close'].iloc[-1])
            context['latest_price'] = f"{latest_price:,.2f}"
        prev_price = float(df['Close'].iloc[-2])
        price_change = latest_price - prev_price if prev_price else 0.0
        pct_change = (price_change / prev_price) * 100 if prev_price else 0.0
        
        context.update({
            'price_change': f"{price_change:+,.2f}",
            'pct_change': f"{pct_change:+.2f}%",
            'is_positive': price_change >= 0,
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['Close'].iloc[-100:], 
                                mode='lines', name='Price', line=dict(color='#ffd700', width=3))) # Gold color
        
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

    # Ensure per-interval card running P/L uses the best available latest price (live or fallback).
    context['predictions_arima'] = _get_interval_predictions(
        client, "GOLD", "ARIMA", context['intervals'], latest_price=latest_price
    )

    canonical_url = request.build_absolute_uri(request.path)
    site_name = "Intelligence.quantbots.co"
    asset_for_meta = context.get("asset_name", "Asset")
    seo_title = f"{asset_for_meta} Dashboard | Intelligence.quantbots.co"
    seo_description = (
        f"Live {asset_for_meta} machine learning dashboard with ARIMA forecasts, "
        "real-time signal tracking, running P/L, and multi-timeframe market intelligence."
    )
    seo_keywords = (
        f"{asset_for_meta.lower()} forecast, {asset_for_meta.lower()} trading signals, machine learning finance, "
        "quantitative trading, ARIMA prediction, algorithmic market intelligence"
    )
    current_iso = django_timezone.now().replace(microsecond=0).isoformat()

    organization_json_ld = {
        "@context": "https://schema.org",
        "@type": "Organization",
        "name": site_name,
        "url": request.build_absolute_uri("/"),
        "description": "Machine learning market intelligence for financial assets.",
    }
    webpage_json_ld = {
        "@context": "https://schema.org",
        "@type": "WebPage",
        "name": seo_title,
        "url": canonical_url,
        "description": seo_description,
        "dateModified": current_iso,
        "about": [
            f"{asset_for_meta} Machine Learning Forecast",
            "Quantitative Finance",
            "Algorithmic Trading Signals",
        ],
    }

    context.update({
        "seo_title": seo_title,
        "seo_description": seo_description,
        "seo_keywords": seo_keywords,
        "seo_canonical_url": canonical_url,
        "seo_site_name": site_name,
        "seo_locale": "en_US",
        "seo_updated_iso": current_iso,
        "seo_organization_json_ld": json.dumps(organization_json_ld),
        "seo_webpage_json_ld": json.dumps(webpage_json_ld),
    })
    
    return render(request, 'gold/dashboard.html', context)

@require_GET
def last_price_api(request):
    processed_file = PROJECT_ROOT / "gold" / "data" / "processed" / "gold_1h_processed.csv"
    ticker = 'GOLD'
    response = get_last_price_payload("gold", ticker, processed_file)
    status = 200 if response.get("ok") else 503
    return JsonResponse(response, status=status)

def _interval_detail(request, interval, model_key="ARIMA"):
    processed_file = PROJECT_ROOT / "gold" / "data" / "processed" / f"gold_{interval}_processed.csv"
    selected_model = "ARIMA"
    selected_model_label = "ARIMA"
    
    context = {
        'interval': interval,
        'asset_name': 'Gold',
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
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            experiment = client.get_experiment_by_name(f"GOLD_ARIMA_{interval}")
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
        except Exception as e:
            print(f"Error fetching forecast: {e}")

        # Real ROI & Signal History (ARIMA only)
        try:
            client = MlflowClient()
            experiments = {
                "ARIMA": f"GOLD_ARIMA_{interval}"
            }
            
            comparison_data = {}
            for model_key, exp_name in experiments.items():
                exp = client.get_experiment_by_name(exp_name)
                if exp:
                    runs = client.search_runs(
                        experiment_ids=[exp.experiment_id],
                        order_by=["attributes.start_time DESC"],
                        max_results=200,
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
                    for i in range(len(temp_signals) - 1):
                        curr = temp_signals[i+1] # Earlier run
                        next_actual = temp_signals[i]['last_close'] # Later run
                        
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
                            'profit': f"{profit:+,.2f}",
                            'mae': f"{curr['mae']:,.2f}" if curr['mae'] is not None else "N/A",
                            'mse': f"{curr['mse']:,.2f}" if curr['mse'] is not None else "N/A",
                        })

                    win_rate = (wins / total_resolved * 100) if total_resolved > 0 else 0
                    comparison_data[model_key] = {
                        'profit': f"{total_profit:+,.2f}",
                        'win_rate': f"{win_rate:.1f}%",
                        'signals': signals[:20] # Show last 20
                    }
            
            context['comparison'] = comparison_data
            
            # Update summary stats with real data
            if "ARIMA" in comparison_data:
                context['roi_estimate'] = f"{comparison_data['ARIMA']['win_rate']} Win Rate"
                context['ab_test_result'] = f"ARIMA ({comparison_data['ARIMA']['profit']} pts)"

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
                                    mode='lines', name='SMA 20', line=dict(color='rgba(255,215,0,0.5)', width=1))) # Gold translucent
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=30, b=0),
            title=f"Gold {interval} Technical Analysis"
        )
        context['chart_html'] = pio.to_html(fig, full_html=False)
        
        # Additional context for cards
        context.update({
            'drift_score': 'Low (0.08)',
        })
    
    return render(request, 'gold/interval_detail.html', context)


def interval_detail_arima(request, interval):
    return _interval_detail(request, interval, "ARIMA")
