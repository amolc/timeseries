from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET
import pandas as pd
import pathlib
import plotly.graph_objects as go
import plotly.io as pio
import os
import json
import mlflow
from django.utils import timezone
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


def _format_duration(start_time_raw, end_time_raw):
    try:
        start_ts = pd.to_datetime(start_time_raw, utc=True)
        end_ts = pd.to_datetime(end_time_raw, utc=True)
    except Exception:
        return "N/A"
    if pd.isna(start_ts) or pd.isna(end_ts):
        return "N/A"
    delta = end_ts - start_ts
    if delta.total_seconds() < 0:
        return "N/A"
    total_minutes = int(delta.total_seconds() // 60)
    days, rem_minutes = divmod(total_minutes, 1440)
    hours, minutes = divmod(rem_minutes, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


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
        'predictions_lr': {},
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
    latest_price_payload = get_last_price_payload("usoil_dashboard", "USOIL/TVC", processed_file)
    if latest_price_payload.get("ok"):
        context['latest_price'] = latest_price_payload.get("latest_price", context["latest_price"])
        context['latest_price_as_of'] = latest_price_payload.get("as_of", context["latest_price_as_of"])
        try:
            latest_price = float(str(context['latest_price']).replace(",", ""))
        except (TypeError, ValueError):
            latest_price = None

    # Pick the latest active call from 1h LR/ARIMA runs (whichever is newest).
    latest_call = None
    for exp_name in ("USOIL_LR_1h", "USOIL_ARIMA_1h"):
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

    context['predictions_lr'] = _get_interval_predictions(
        client, "USOIL", "LR", context['intervals'], latest_price=latest_price
    )
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
    context['predictions_lr'] = _get_interval_predictions(
        client, "USOIL", "LR", context['intervals'], latest_price=latest_price
    )
    context['predictions_arima'] = _get_interval_predictions(
        client, "USOIL", "ARIMA", context['intervals'], latest_price=latest_price
    )

    canonical_url = request.build_absolute_uri(request.path)
    site_name = "Intelligence.quantbots.co"
    asset_for_meta = context.get("asset_name", "Asset")
    seo_title = f"{asset_for_meta} Dashboard | Intelligence.quantbots.co"
    seo_description = (
        f"Live {asset_for_meta} machine learning dashboard with Linear Regression and ARIMA forecasts, "
        "real-time signal tracking, running P/L, and multi-timeframe market intelligence."
    )
    seo_keywords = (
        f"{asset_for_meta.lower()} forecast, {asset_for_meta.lower()} trading signals, machine learning finance, "
        "quantitative trading, ARIMA prediction, linear regression forecast, algorithmic market intelligence"
    )
    current_iso = timezone.now().replace(microsecond=0).isoformat()

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


def _interval_detail(request, interval, model_override=None):
    selected_raw = (model_override or request.GET.get("model") or "ALL").strip().upper()
    if selected_raw in ("LINEAR", "LINEAR_REGRESSION"):
        selected_raw = "LR"
    selected_model = selected_raw if selected_raw in {"ALL", "LR", "ARIMA"} else "ALL"
    selected_model_label = {
        "ALL": "All Models",
        "LR": "Linear Regression",
        "ARIMA": "ARIMA",
    }[selected_model]

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
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiments = {
            "LR": f"USOIL_LR_{interval}",
            "ARIMA": f"USOIL_ARIMA_{interval}",
        }

        comparison_data = {
            "LR": {"profit": "N/A", "win_rate": "N/A", "signals": [], "changeover_signals": [], "runs": [], "run_count": 0, "curve": []},
            "ARIMA": {"profit": "N/A", "win_rate": "N/A", "signals": [], "changeover_signals": [], "runs": [], "run_count": 0, "curve": []},
        }

        for model_key, exp_name in experiments.items():
            exp = client.get_experiment_by_name(exp_name)
            if not exp:
                continue

            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=200,
            )
            if not runs:
                continue

            wins = 0
            total_resolved = 0
            total_profit = 0.0
            signals = []
            temp_signals = []
            run_rows = []

            for run in runs:
                last_close = _get_last_close_price(run)
                pred_next = _get_predicted_price(run)
                run_time = run.data.params.get("last_record_time", "N/A")

                if last_close is not None and pred_next is not None:
                    delta = pred_next - last_close
                    delta_pct = (delta / last_close * 100) if last_close else 0.0
                    signal = "BUY" if pred_next > last_close else "SELL"
                    temp_signals.append({
                        "time": run_time,
                        "last_close": last_close,
                        "pred_next": pred_next,
                        "signal": signal,
                        "run_id": run.info.run_id[:8],
                    })
                    run_rows.append({
                        "time": run_time,
                        "last_close": f"{last_close:,.2f}",
                        "pred_next": f"{pred_next:,.2f}",
                        "actual_next": "N/A",
                        "delta": f"{delta:+,.2f}",
                        "delta_pct": f"{delta_pct:+.2f}%",
                        "signal": signal,
                        "signal_class": "success" if signal == "BUY" else "danger",
                        "run_id": run.info.run_id[:8],
                    })

            # Newest run has no realized actual yet; older runs use the next observed close.
            for idx in range(1, len(temp_signals)):
                run_rows[idx]["actual_next"] = f"{temp_signals[idx - 1]['last_close']:,.2f}"

            # Include the newest run in signal history as PENDING when actual is not available yet.
            for idx, curr in enumerate(temp_signals):
                baseline_close = curr["last_close"]
                delta = curr["pred_next"] - baseline_close
                delta_pct = (delta / baseline_close * 100) if baseline_close else 0.0
                signal_row = {
                    "time": curr["time"],
                    "last_close": f"{baseline_close:,.2f}",
                    "pred_next": f"{curr['pred_next']:,.2f}",
                    "predicted": f"{curr['pred_next']:,.2f}",
                    "delta": f"{delta:+,.2f}",
                    "delta_pct": f"{delta_pct:+.2f}%",
                    "signal": curr["signal"],
                    "signal_class": "success" if curr["signal"] == "BUY" else "danger",
                    "run_id": curr["run_id"],
                }

                if idx == 0:
                    signal_row.update({
                        "actual_next": "N/A",
                        "actual_delta": "N/A",
                        "result": "PENDING",
                        "result_class": "warning",
                        "profit": "N/A",
                    })
                    signals.append(signal_row)
                    continue

                next_actual = temp_signals[idx - 1]["last_close"]
                actual_delta = next_actual - curr["last_close"]
                points = abs(next_actual - curr["last_close"])
                if points == 0:
                    outcome = "DRAW"
                    outcome_class = "secondary"
                    profit = 0.0
                else:
                    if curr["signal"] == "BUY":
                        is_win = next_actual > curr["last_close"]
                    else:
                        is_win = next_actual < curr["last_close"]
                    outcome = "WIN" if is_win else "LOSS"
                    outcome_class = "success" if is_win else "danger"
                    profit = points if is_win else -points
                    wins += 1 if is_win else 0
                    total_resolved += 1

                total_profit += profit

                signal_row.update({
                    "actual_next": f"{next_actual:,.2f}",
                    "actual_delta": f"{actual_delta:+,.2f}",
                    "result": outcome,
                    "result_class": outcome_class,
                    "profit": f"{profit:+,.2f}",
                })
                signals.append(signal_row)

            # Simulate crossover trading: enter on trigger close, exit on next opposite trigger close.
            crossover_chain = []
            for row in reversed(temp_signals):  # oldest -> newest
                if not crossover_chain or row["signal"] != crossover_chain[-1]["signal"]:
                    crossover_chain.append(row)

            crossover_signals_chrono = []
            crossover_wins = 0
            crossover_resolved = 0
            crossover_total_profit = 0.0
            cumulative_profit = 0.0
            curve = [{"time": "Start", "value": 0.0}]

            for idx, trig in enumerate(crossover_chain):
                entry_price = trig["last_close"]
                pred_next = trig["pred_next"]
                delta = pred_next - entry_price
                delta_pct = (delta / entry_price * 100) if entry_price else 0.0
                row = {
                    "start_time": trig["time"],
                    "run_id": trig["run_id"],
                    "start_close": f"{entry_price:,.2f}",
                    "pred_next": f"{pred_next:,.2f}",  # kept for compatibility in other views
                    "delta": f"{delta:+,.2f}",         # kept for compatibility in other views
                    "delta_pct": f"{delta_pct:+.2f}%", # kept for compatibility in other views
                    "signal": trig["signal"],
                    "signal_class": "success" if trig["signal"] == "BUY" else "danger",
                }

                if idx + 1 < len(crossover_chain):
                    nxt = crossover_chain[idx + 1]
                    exit_price = nxt["last_close"]
                    actual_delta = exit_price - entry_price
                    pnl = (exit_price - entry_price) if trig["signal"] == "BUY" else (entry_price - exit_price)
                    points = abs(exit_price - entry_price)
                    if pnl > 0:
                        outcome = "WIN"
                        outcome_class = "success"
                        crossover_wins += 1
                    elif pnl < 0:
                        outcome = "LOSS"
                        outcome_class = "danger"
                    else:
                        outcome = "DRAW"
                        outcome_class = "secondary"

                    crossover_resolved += 1
                    crossover_total_profit += pnl
                    cumulative_profit += pnl
                    curve.append({"time": nxt["time"], "value": cumulative_profit})

                    row.update({
                        "end_time": nxt["time"],
                        "end_price": f"{exit_price:,.2f}",
                        "duration": _format_duration(trig["time"], nxt["time"]),
                        "profitloss": f"{pnl:+,.2f}",
                        "points": f"{points:,.2f}",
                        "actual_next": f"{exit_price:,.2f}",  # compatibility
                        "actual_delta": f"{actual_delta:+,.2f}",
                        "profit": f"{pnl:+,.2f}",             # compatibility
                        "result": outcome,
                        "result_class": outcome_class,
                    })
                else:
                    row.update({
                        "end_time": "N/A",
                        "end_price": "N/A",
                        "duration": "N/A",
                        "profitloss": "N/A",
                        "points": "N/A",
                        "actual_next": "N/A",
                        "actual_delta": "N/A",
                        "profit": "N/A",
                        "result": "PENDING",
                        "result_class": "warning",
                    })
                    curve.append({"time": trig["time"], "value": cumulative_profit})

                crossover_signals_chrono.append(row)

            changeover_signals = list(reversed(crossover_signals_chrono))  # newest first for display
            win_rate = (crossover_wins / crossover_resolved * 100) if crossover_resolved else 0.0

            comparison_data[model_key] = {
                "profit": f"{crossover_total_profit:+,.2f}",
                "win_rate": f"{win_rate:.1f}%",
                "signals": signals,
                "changeover_signals": changeover_signals,
                "runs": run_rows,
                "run_count": len(run_rows),
                "curve": curve,
            }

            if run_rows:
                context["comparison"].setdefault(model_key, {})
                context["comparison"][model_key]["last_run_time"] = run_rows[0]["time"]
                context["comparison"][model_key]["forecast_price"] = run_rows[0]["pred_next"]

        context["comparison"] = comparison_data

        primary_model = "LR"
        if selected_model in {"LR", "ARIMA"}:
            primary_model = selected_model
        elif comparison_data["LR"]["run_count"] == 0 and comparison_data["ARIMA"]["run_count"] > 0:
            primary_model = "ARIMA"

        if comparison_data[primary_model]["run_count"] > 0:
            first_run = comparison_data[primary_model]["runs"][0]
            context["forecast_price"] = first_run["pred_next"]
            context["forecast_signal"] = first_run["signal"]
            context["forecast_signal_class"] = first_run["signal_class"]
            context["last_run_time"] = first_run["time"]
            context["roi_estimate"] = f"{comparison_data[primary_model]['win_rate']} Win Rate"
            context["running_signal_label"] = f"{first_run['signal']} (Live)"
            context["running_signal_class"] = first_run["signal_class"]
            context["running_call_side"] = first_run["signal"]
            context["running_call_value"] = first_run["last_close"]
            context["running_call_time"] = first_run["time"]
            try:
                trigger_price = float(str(first_run["last_close"]).replace(",", ""))
                running_pnl = (latest_price - trigger_price) if first_run["signal"] == "BUY" else (trigger_price - latest_price)
                if running_pnl > 0:
                    pnl_class = "success"
                elif running_pnl < 0:
                    pnl_class = "danger"
                else:
                    pnl_class = "secondary"
                context["running_profit"] = f"{running_pnl:+,.2f}"
                context["running_profit_class"] = pnl_class
            except (TypeError, ValueError):
                pass

        context["ab_test_result"] = (
            f"LR ({comparison_data['LR']['profit']} pts) vs "
            f"ARIMA ({comparison_data['ARIMA']['profit']} pts)"
        )

        roi_fig = go.Figure()
        roi_model_key = primary_model
        roi_curve = comparison_data.get(roi_model_key, {}).get("curve", [])
        roi_color = "#22c55e" if roi_model_key == "LR" else "#3b82f6"
        context["roi_chart_label"] = "Linear Regression" if roi_model_key == "LR" else "ARIMA"
        if roi_curve:
            roi_fig.add_trace(go.Scatter(
                x=[p["time"] for p in roi_curve],
                y=[p["value"] for p in roi_curve],
                mode="lines+markers",
                name=context["roi_chart_label"],
                line=dict(color=roi_color, width=2),
                marker=dict(size=4),
            ))
        if roi_fig.data:
            roi_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=8, b=0),
                height=130,
                showlegend=False,
                yaxis=dict(title="Cumulative Points"),
                xaxis=dict(showticklabels=False),
            )
            context["roi_chart_html"] = pio.to_html(
                roi_fig,
                full_html=False,
                config={"responsive": True, "displayModeBar": False},
            )
    except Exception as e:
        print(f"Error fetching ROI data: {e}")

    return render(request, "usoil/interval_detail.html", context)


def interval_detail(request, interval):
    return _interval_detail(request, interval)


def interval_detail_lr(request, interval):
    return _interval_detail(request, interval, "LR")


def interval_detail_arima(request, interval):
    return _interval_detail(request, interval, "ARIMA")
