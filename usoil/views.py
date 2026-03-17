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
import mlflow
from mlflow.tracking import MlflowClient
from django.utils import timezone as django_timezone
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


def _format_duration(start_time_raw, end_time_raw):
    """Robust duration calculation using pandas and timezone-aware datetimes."""
    try:
        start_ts = pd.to_datetime(start_time_raw, utc=True)
        end_ts = pd.to_datetime(end_time_raw, utc=True)
        diff = end_ts - start_ts
        
        total_seconds = int(diff.total_seconds())
        if total_seconds < 0:
            return "---"
            
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, _ = divmod(remainder, 60)
        
        parts = []
        if days > 0: parts.append(f"{days}d")
        if hours > 0: parts.append(f"{hours}h")
        if minutes >= 0: parts.append(f"{minutes}m")
        
        return " ".join(parts) if parts else "0m"
    except Exception:
        return "N/A"


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


def usoil_dashboard(request):
    """Main USOIL Dashboard with interval selection cards and top predictions."""
    processed_file = PROJECT_ROOT / "usoil" / "data" / "processed" / "usoil_1h_processed.csv"

    context = {
        'asset_name': 'USOIL',
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

    client = MlflowClient()

    latest_price = None
    latest_price_payload = get_last_price_payload("usoil_dashboard", "USOIL/TVC", processed_file)
    if latest_price_payload.get("ok"):
        context['latest_price'] = latest_price_payload.get("latest_price", context["latest_price"])
        context['latest_price_as_of'] = latest_price_payload.get("as_of", context["latest_price_as_of"])
        try:
            latest_price = float(str(context['latest_price']).replace(",", ""))
        except (TypeError, ValueError):
            latest_price = None

    # Pick the latest active call from 1h ARIMA runs.
    latest_call = None
    for exp_name in ("USOIL_ARIMA_1h",):
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
        client, "USOIL", "ARIMA", context['intervals'], latest_price=latest_price
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

    # Ensure per-interval card running P/L uses the best available latest price (live or fallback).
    context['predictions_arima'] = _get_interval_predictions(
        client, "USOIL", "ARIMA", context['intervals'], latest_price=latest_price
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

    return render(request, 'usoil/dashboard.html', context)


@require_GET
def last_price_api(request):
    processed_file = PROJECT_ROOT / "usoil" / "data" / "processed" / "usoil_1h_processed.csv"
    ticker = 'USOIL/TVC'
    response = get_last_price_payload("usoil", ticker, processed_file)
    status = 200 if response.get("ok") else 503
    return JsonResponse(response, status=status)


def _interval_detail(request, interval, model_override="ARIMA"):
    selected_model = "ARIMA"
    selected_model_label = "ARIMA"

    processed_file = PROJECT_ROOT / "usoil" / "data" / "processed" / f"usoil_{interval}_processed.csv"
    context = {
        "interval": interval,
        "asset_name": "USOIL",
        "available_intervals": ["1h", "1d", "1w", "1m"],
        "forecast_price": "N/A",
        "forecast_signal": "N/A",
        "forecast_signal_class": "secondary",
        "last_run_time": "N/A",
        "latest_price": "N/A",
        "latest_price_change": "N/A",
        "latest_price_pct": "N/A",
        "latest_price_is_positive": True,
        "latest_price_as_of": "N/A",
        "running_signal_label": "N/A",
        "running_signal_class": "secondary",
        "running_call_value": "N/A",
        "running_call_time": "N/A",
        "running_call_side": "N/A",
        "running_profit": "N/A",
        "running_profit_class": "secondary",
        "selected_model": selected_model,
        "selected_model_label": selected_model_label,
        "chart_html": "",
        "comparison": {},
        "completed_runs": [],
        "drift_score": "Stable",
        "ab_test_result": "No recent runs yet",
        "roi_estimate": "N/A",
        "roi_chart_html": "",
        "roi_chart_label": "",
    }

    if not processed_file.exists():
        return render(request, "usoil/interval_detail.html", context)

    df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
    if df.empty:
        return render(request, "usoil/interval_detail.html", context)

    # Use the same live/fallback payload as the public API so header price stays current.
    latest_price_payload = get_last_price_payload("usoil_interval_detail", "USOIL/TVC", processed_file)
    if latest_price_payload.get("ok"):
        context["latest_price"] = latest_price_payload.get("latest_price", context["latest_price"])
        context["latest_price_change"] = latest_price_payload.get("price_change", context["latest_price_change"])
        context["latest_price_pct"] = latest_price_payload.get("pct_change", context["latest_price_pct"])
        context["latest_price_is_positive"] = bool(latest_price_payload.get("is_positive", True))
        context["latest_price_as_of"] = latest_price_payload.get("as_of", context["latest_price_as_of"])
        try:
            latest_price = float(str(context["latest_price"]).replace(",", ""))
        except (TypeError, ValueError):
            latest_price = float(df["Close"].iloc[-1])
            context["latest_price"] = f"{latest_price:,.2f}"
    else:
        latest_price = float(df["Close"].iloc[-1])
        context["latest_price"] = f"{latest_price:,.2f}"

    chart_window = min(len(df), 300)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index[-chart_window:],
        open=df["Open"].iloc[-chart_window:],
        high=df["High"].iloc[-chart_window:],
        low=df["Low"].iloc[-chart_window:],
        close=df["Close"].iloc[-chart_window:],
        name="Market Data",
    ))
    if "MA7" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index[-chart_window:],
            y=df["MA7"].iloc[-chart_window:],
            mode="lines",
            name="MA7",
            line=dict(color="rgba(255,255,255,0.5)", width=1),
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    context["chart_html"] = pio.to_html(fig, full_html=False)

    try:
        client = MlflowClient()
        # Real ROI & Model (ARIMA Only)
        experiments = {
            "ARIMA": f"USOIL_ARIMA_{interval}"
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
                    
                    # Convert runs to signals for processing
                    temp_signals = []
                    for run in runs:
                        last_close = _get_last_close_price(run)
                        pred_next = _get_predicted_price(run)
                        run_time = _fmt_run_timestamp(run.info.start_time)
                        metrics = run.data.metrics
                        
                        if last_close is not None and pred_next is not None:
                            signal = "BUY" if pred_next > last_close else "SELL"
                            # Prefer manual timestamp from params, fallback to run start
                            run_time = run.data.params.get('last_record_time')
                            if not run_time or run_time == "N/A":
                                run_time = _fmt_run_timestamp(run.info.start_time)
                                
                            temp_signals.append({
                                'time': run_time,
                                'last_close': last_close,
                                'pred_next': pred_next,
                                'signal': signal,
                                'run_id': run.info.run_id[:8],
                                'mae': metrics.get("mae"),
                                'mse': metrics.get("mse"),
                            })
                    
                    # Process signals to determine wins/losses
                    full_history = []
                    wins = 0
                    total_resolved = 0
                    total_profit = 0.0
                    
                    for idx, curr in enumerate(temp_signals):
                        baseline_close = curr["last_close"]
                        
                        signal_row = {
                            "time": curr["time"],
                            "run_id": curr["run_id"],
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
                            full_history.append(signal_row)
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
                        full_history.append(signal_row)

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
                    
                    # 2. Format Changeovers for display
                    formatted_changeovers = []
                    
                    for i in range(len(changeovers)):
                        co = changeovers[i]
                        row = {
                            "time": co["time"],
                            "start_time": co["time"],
                            "signal": co["signal"],
                            "signal_class": "success" if co["signal"] == "BUY" else "danger",
                            "last_close": f"{co['last_close']:,.2f}",
                            "start_close": f"{co['last_close']:,.2f}",
                            "end_time": "Active",
                            "end_price": "---",
                            "duration": "---",
                            "profit_loss": "---",
                            "profitloss": "---",
                            "pnl_class": "muted",
                            "result": "RUNNING",
                            "result_class": "warning",
                            "run_id": co["run_id"],
                        }
                        
                        if i > 0:
                            newer_co = changeovers[i-1]
                            row["end_time"] = newer_co["time"]
                            row["end_price"] = f"{newer_co['last_close']:,.2f}"
                            row["duration"] = _format_duration(co["time"], newer_co["time"])
                            
                            # Next-trade-start-price-based P/L
                            pnl = (newer_co["last_close"] - co["last_close"]) if co["signal"] == "BUY" else (co["last_close"] - newer_co["last_close"])
                            row["profit_loss"] = f"{pnl:+,.2f}"
                            row["profitloss"] = f"{pnl:+,.2f}"
                            row["profit_loss_val"] = pnl
                            row["pnl_class"] = "success" if pnl > 0 else "danger"
                            row["result"] = "WIN" if pnl > 0 else "LOSS"
                            row["result_class"] = "success" if pnl > 0 else "danger"
                        else:
                            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                            row["duration"] = _format_duration(co["time"], now_str)
                            
                        formatted_changeovers.append(row)

                    # 3. Create ROI Chart (Equity Curve)
                    sig_history = [s for s in formatted_changeovers if s["result"] != "RUNNING" and s["result"] != "OPEN"]
                    roi_chart_html = ""
                    if sig_history:
                        sig_history.reverse() # Chronological
                        equity = 0
                        x_vals = []
                        y_vals = []
                        for s in sig_history:
                            equity += s.get("profit_loss_val", 0)
                            x_vals.append(s["time"])
                            y_vals.append(equity)

                        roi_fig = go.Figure()
                        roi_fig.add_trace(go.Scatter(
                            x=x_vals, 
                            y=y_vals,
                            mode='lines',
                            name='Equity Curve',
                            line=dict(color='#00ff00' if equity >= 0 else '#ff4d4d', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(0, 255, 0, 0.1)' if equity >= 0 else 'rgba(255, 0, 0, 0.1)'
                        ))
                        roi_fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=0, r=0, t=10, b=0),
                            height=120,
                            xaxis=dict(showgrid=False, visible=False),
                            yaxis=dict(showgrid=True, gridcolor='#333', visible=False),
                        )
                        roi_chart_html = pio.to_html(roi_fig, full_html=False, config={'displayModeBar': False})

                    win_rate = (wins / total_resolved * 100) if total_resolved > 0 else 0
                    comparison_data[model_key] = {
                        'profit': f"{total_profit:+,.2f}",
                        'profit_val': total_profit,
                        'win_rate': f"{win_rate:.1f}%",
                        'signals': full_history[:20],
                        'changeover_signals': formatted_changeovers,
                        'roi_chart': roi_chart_html
                    }
            
        if selected_model in comparison_data:
            context['comparison'] = {selected_model: comparison_data[selected_model]}
            
            # Update top-level context with the most recent run data
            if comparison_data[selected_model]['signals']:
                latest_sig = comparison_data[selected_model]['signals'][0]
                context['forecast_price'] = latest_sig['pred_next']
                context['forecast_signal'] = latest_sig['signal']
                context['forecast_signal_class'] = latest_sig['signal_class']
                context['last_run_time'] = latest_sig['time']
                
                # Update running stats
                context['running_signal_label'] = f"{latest_sig['signal']} (Live)"
                context['running_signal_class'] = latest_sig['signal_class']
                context['running_call_side'] = latest_sig['signal']
                context['running_call_value'] = latest_sig['last_close']
                context['running_call_time'] = latest_sig['time']
                
                try:
                    trigger_price = float(str(latest_sig['last_close']).replace(",", ""))
                    current_pnl = (latest_price - trigger_price) if latest_sig['signal'] == "BUY" else (trigger_price - latest_price)
                    context['running_profit'] = f"{current_pnl:+,.2f}"
                    context['running_profit_class'] = "success" if current_pnl > 0 else "danger"
                except:
                    pass

        # Summary stats for the page
        if selected_model in comparison_data:
            context['roi_estimate'] = f"{comparison_data[selected_model]['win_rate']} Win Rate"
            context['ab_test_result'] = f"{selected_model} ({comparison_data[selected_model]['profit']} pts)"
            context['roi_chart_html'] = comparison_data[selected_model]['roi_chart']

    except Exception as e:
        print(f"Error fetching ROI data: {e}")
        context['comparison'] = {}

    return render(request, "usoil/interval_detail.html", context)


def interval_detail(request, interval):
    return _interval_detail(request, interval)


def interval_detail_arima(request, interval):
    return _interval_detail(request, interval, "ARIMA")
