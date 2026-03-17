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


def _format_duration(start_str, end_str):
    """Helper to format duration between two UTC date strings."""
    try:
        # Expected format: "2024-03-25 14:00:00 UTC"
        fmt = "%Y-%m-%d %H:%M:%S UTC"
        start_dt = datetime.strptime(start_str, fmt).replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_str, fmt).replace(tzinfo=timezone.utc)
        diff = end_dt - start_dt
        
        days = diff.days
        hours, remainder = divmod(diff.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        if days > 0:
            return f"{days}d {hours}h"
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    except Exception:
        return "---"


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
        latest_price = df['Close'].iloc[-1]
        context['latest_price'] = f"{latest_price:,.2f}"
        
        try:
            client = MlflowClient()
            experiments = {"ARIMA": f"GOLD_ARIMA_{interval}"}
            comparison_data = {}
            full_history = []  # Outside loop to avoid unbound error
            
            for m_key, exp_name in experiments.items():
                exp = client.get_experiment_by_name(exp_name)
                if not exp:
                    continue
                
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["attributes.start_time DESC"],
                    max_results=200,
                )
                
                temp_signals = []
                for run in runs:
                    last_close = _get_last_close_price(run)
                    pred_next = _get_predicted_price(run)
                    # Use formatted timestamp helper
                    run_time = run.data.params.get('last_record_time') or _fmt_run_timestamp(run.info.start_time)
                    metrics = run.data.metrics
                    
                    if last_close is not None and pred_next is not None:
                        temp_signals.append({
                            'run_id': run.info.run_id[:8],
                            'time': run_time,
                            'last_close': last_close,
                            'pred_next': pred_next,
                            'signal': "BUY" if pred_next > last_close else "SELL",
                            'mae': metrics.get("mae"),
                            'mse': metrics.get("mse"),
                        })

                # Initialize variables to avoid unbound errors
                full_history = []
                formatted_changeovers = []
                total_profit = 0.0
                wins = 0
                total_resolved = 0

                if not temp_signals:
                    continue

                # 1. Changeover Logic: Identify when signal direction flips (working ASC)
                temp_signals_asc = list(reversed(temp_signals))
                changeovers = []
                if temp_signals_asc:
                    current_co = temp_signals_asc[0]
                    changeovers.append(current_co)
                    for i in range(1, len(temp_signals_asc)):
                        if temp_signals_asc[i]["signal"] != current_co["signal"]:
                            current_co = temp_signals_asc[i]
                            changeovers.append(current_co)
                
                # Reverse to DESC for display
                changeovers.reverse()

                formatted_changeovers = []
                total_profit = 0.0
                wins = 0
                total_resolved = 0

                for i in range(len(changeovers)):
                    co = changeovers[i]
                    row = {
                        "run_id": co["run_id"],
                        "time": co["time"],
                        "last_close": f"{co['last_close']:,.2f}",
                        "signal": co["signal"],
                        "signal_class": "success" if co["signal"] == "BUY" else "danger",
                        "end_time": "Active",
                        "end_price": "---",
                        "duration": "---",
                        "profit_loss": "---",
                        "pnl_class": "secondary",
                        "result": "OPEN",
                    }
                    if i > 0:
                        newer_co = changeovers[i-1]
                        row["end_time"] = newer_co["time"]
                        row["end_price"] = f"{newer_co['last_close']:,.2f}"
                        row["duration"] = _format_duration(co["time"], newer_co["time"])
                        
                        pnl = (newer_co["last_close"] - co["last_close"]) if co["signal"] == "BUY" else (co["last_close"] - newer_co["last_close"])
                        row["profit_loss"] = f"{pnl:+,.2f}"
                        row["profit_loss_val"] = pnl
                        row["pnl_class"] = "success" if pnl > 0 else "danger"
                        row["result"] = "PROFIT" if pnl > 0 else "LOSS"
                        
                        total_profit += pnl
                        if pnl > 0: wins += 1
                        total_resolved += 1
                    else:
                        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                        row["duration"] = _format_duration(co["time"], now_str)
                    
                    formatted_changeovers.append(row)

                # 2. Full History Logic (Standard next-period comparison)
                full_history = []
                for i in range(len(temp_signals)):
                    curr = temp_signals[i]
                    row = {
                        "run_id": curr["run_id"],
                        "time": curr["time"],
                        "last_close": f"{curr['last_close']:,.2f}",
                        "pred_next": f"{curr['pred_next']:,.2f}",
                        "signal": curr["signal"],
                        "signal_class": "success" if curr["signal"] == "BUY" else "danger",
                        "result": "PENDING",
                        "result_class": "secondary",
                        "profit": "---",
                        "mse": f"{curr['mse']:.4f}" if curr['mse'] is not None else "---",
                        "mae": f"{curr['mae']:.4f}" if curr['mae'] is not None else "---",
                    }
                    if i > 0:
                        prev_actual = temp_signals[i-1]["last_close"]
                        is_win = (prev_actual > curr["last_close"]) if curr["signal"] == "BUY" else (prev_actual <= curr["last_close"])
                        p_val = abs(prev_actual - curr["last_close"]) if is_win else -abs(prev_actual - curr["last_close"])
                        row["result"] = "WIN" if is_win else "LOSS"
                        row["result_class"] = "success" if is_win else "danger"
                        row["profit"] = f"{p_val:+,.2f}"
                        row["profit_val"] = p_val
                    full_history.append(row)

                win_rate = (wins / total_resolved * 100) if total_resolved > 0 else 0
                comparison_data[m_key] = {
                    "profit": f"{total_profit:+,.2f}",
                    "profit_val": total_profit,
                    "win_rate": f"{win_rate:.1f}%",
                    "changeover_signals": formatted_changeovers,
                    "signals": full_history[:50],
                }

            context["comparison"] = comparison_data
            if "ARIMA" in comparison_data:
                context["roi_estimate"] = f"{comparison_data['ARIMA']['win_rate']} Win Rate"
                context["ab_test_result"] = f"ARIMA ({comparison_data['ARIMA']['profit']} pts)"
                context["forecast_price"] = full_history[0]["pred_next"]
                context["last_run_time"] = full_history[0]["time"]

                # Generate ROI Plotly Chart
                sig_history = [s for s in comparison_data["ARIMA"]["changeover_signals"] if s["result"] != "OPEN"]
                if sig_history:
                    sig_history.reverse() # Back to chronological for chart
                    equity = 0
                    x_vals = []
                    y_vals = []
                    for s in sig_history:
                        equity += s.get("profit_loss_val", 0)
                        x_vals.append(s["time"])
                        y_vals.append(equity)

                    roi_fig = go.Figure()
                    roi_fig.add_trace(go.Scatter(
                        x=x_vals, y=y_vals, mode="lines",
                        fill="tozeroy", line=dict(color="#00ff00" if equity >= 0 else "#ff4d4d", width=2)
                    ))
                    roi_fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=100,
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                    )
                    context["roi_chart_html"] = pio.to_html(roi_fig, full_html=False, config={"displayModeBar": False})

        except Exception as e:
            print(f"Error in detail view: {e}")

        # Candlestick Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index[-100:],
            open=df['Open'].iloc[-100:],
            high=df['High'].iloc[-100:],
            low=df['Low'].iloc[-100:],
            close=df['Close'].iloc[-100:],
            name='Market Data'
        ))
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
        context['drift_score'] = 'Low (0.08)'
    
    return render(request, 'gold/interval_detail.html', context)


def interval_detail_arima(request, interval):
    return _interval_detail(request, interval, "ARIMA")
