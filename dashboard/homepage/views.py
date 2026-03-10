import os
import json
import pathlib
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from django.shortcuts import render
from django.conf import settings
from django.utils import timezone
import plotly.graph_objects as go
from plotly.offline import plot
from .utils import get_landing_assets_data
from utils.live_price import get_last_price_payload


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


def _get_latest_prediction_for_experiment(client, experiment_name):
    try:
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            return None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None
        return _get_predicted_price(runs[0])
    except Exception:
        return None


def _get_latest_run_snapshot(client, experiment_name):
    try:
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            return None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None
        run = runs[0]
        pred = _get_predicted_price(run)
        last_close = _get_last_close_price(run)
        if pred is None or last_close is None:
            return None
        signal = "BUY" if pred > last_close else "SELL"
        return {
            "pred": pred,
            "last_close": last_close,
            "signal": signal,
            "signal_class": "success" if signal == "BUY" else "danger",
            "run_time": run.data.params.get("last_record_time", "N/A"),
        }
    except Exception:
        return None


def landing_page(request):
    """
    Enhanced landing page with 3-month interactive data for BTC, GOLD, SPX500, and NIFTY.
    """
    # Configure MLflow for BTC prediction from the new central DB
    try:
        project_root = settings.BASE_DIR.parent
        mlflow_db_path = os.path.join(project_root, "mlflow.db")
        tracking_uri = f"sqlite:///{mlflow_db_path}"
        mlflow.set_tracking_uri(tracking_uri)
    except Exception:
        pass
    
    client = MlflowClient()
    latest_prediction_arima_1h = "N/A"
    
    arima_prediction_map = {}
    arima_snapshot_map = {}
    for asset_name in ("BTCUSD", "PAXUSD", "SPX500", "GOLD", "NIFTY", "USOIL"):
        snapshot = _get_latest_run_snapshot(client, f"{asset_name}_ARIMA_1h")
        if snapshot:
            arima_snapshot_map[asset_name] = snapshot
            arima_prediction_map[asset_name] = f"{snapshot['pred']:,.2f}"
    latest_prediction_arima_1h = arima_prediction_map.get("BTCUSD", "N/A")

    project_root = pathlib.Path(settings.BASE_DIR).parent
    btc_processed = project_root / "btcusd" / "data" / "processed" / "btcusd_1h_processed.csv"
    btc_price_payload = get_last_price_payload("homepage_btc", "BTC-USD", btc_processed)

    btc_latest_price = "N/A"
    btc_latest_change = "N/A"
    btc_latest_pct_change = "N/A"
    btc_is_positive = True
    btc_as_of = "N/A"
    if btc_price_payload.get("ok"):
        btc_latest_price = btc_price_payload.get("latest_price", "N/A")
        btc_latest_change = btc_price_payload.get("price_change", "N/A")
        btc_latest_pct_change = btc_price_payload.get("pct_change", "N/A")
        btc_is_positive = bool(btc_price_payload.get("is_positive", True))
        btc_as_of = btc_price_payload.get("as_of", "N/A")

    assets_data = get_landing_assets_data()
    # Add PAXUSD to assets if needed, though get_landing_assets_data might already handle it
    # For now, let's ensure PAXUSD is in the list if we want it on the landing page
    if 'PAXUSD' not in assets_data:
        # This is just a safeguard, ideally get_landing_assets_data should be updated
        pass
    
    asset_live_config = {
        "BTCUSD": {"ticker": "BTC-USD", "processed": project_root / "btcusd" / "data" / "processed" / "btcusd_1h_processed.csv"},
        "PAXUSD": {"ticker": "PAXG-USD", "processed": project_root / "paxusd" / "data" / "processed" / "paxusd_1h_processed.csv"},
        "SPX500": {"ticker": "^GSPC", "processed": project_root / "spx500" / "data" / "processed" / "spx500_1h_processed.csv"},
        "GOLD": {"ticker": "GC=F", "processed": project_root / "gold" / "data" / "processed" / "gold_1h_processed.csv"},
        "NIFTY": {"ticker": "^NSEI", "processed": project_root / "nifty" / "data" / "processed" / "nifty_1h_processed.csv"},
        "USOIL": {"ticker": "USOIL/TVC", "processed": project_root / "usoil" / "data" / "processed" / "usoil_1h_processed.csv"},
    }

    live_payloads = {}
    for asset_name, cfg in asset_live_config.items():
        live_payloads[asset_name] = get_last_price_payload(
            f"homepage_{asset_name.lower()}",
            cfg["ticker"],
            cfg["processed"],
        )

    asset_info = []
    
    for name, df in assets_data.items():
        if df is not None and not df.empty and len(df) >= 2:
            # Re-ensure index is datetime and sorted
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Ensure we have a single Series for 'Close'
            close_series = df['Close']
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
            
            latest_price = float(close_series.iloc[-1])
            prev_price = float(close_series.iloc[-2])
            price_change = latest_price - prev_price
            pct_change = (price_change / prev_price) * 100 if prev_price else 0.0
            as_of = pd.to_datetime(df.index[-1]).strftime("%Y-%m-%d %H:%M:%S")

            # Prefer latest intraday snapshot for cards if available.
            live_payload = live_payloads.get(name, {})
            if live_payload.get("ok"):
                try:
                    latest_price = float(str(live_payload.get("latest_price", "")).replace(",", ""))
                    price_change = float(str(live_payload.get("price_change", "")).replace(",", ""))
                    pct_change_text = str(live_payload.get("pct_change", ""))
                    pct_change = float(pct_change_text.replace("%", "").replace(",", ""))
                    as_of = str(live_payload.get("as_of", as_of))
                except (TypeError, ValueError):
                    pass

            # Add technical indicators if needed (e.g., MA7)
            if len(df) >= 7:
                df['MA7'] = close_series.rolling(window=7).mean()
            
            # Generate mini-graph for the card
            fig_mini = go.Figure()
            fig_mini.add_trace(go.Scatter(
                x=df.index, 
                y=close_series, 
                mode='lines',
                line=dict(color='#f7931a' if pct_change >= 0 else '#ff4d4d', width=2),
                fill='tozeroy',
                fillcolor='rgba(247, 147, 26, 0.1)' if pct_change >= 0 else 'rgba(255, 77, 77, 0.1)',
            ))
            fig_mini.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(l=0, r=0, t=0, b=0),
                height=60,
                showlegend=False,
                hovermode=False
            )
            mini_plot = plot(fig_mini, output_type='div', include_plotlyjs=False, show_link=False, config={'displayModeBar': False})
            
            # Main interactive graph for the modal/module
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(
                x=df.index, 
                y=close_series, 
                mode='lines', 
                name=f'{name} Price',
                line=dict(color='#f7931a', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(247, 147, 26, 0.05)',
                hovertemplate='<b>Price:</b> $%{y:,.2f}<br><b>Date:</b> %{x}<extra></extra>'
            ))
            
            # Add BTC forecast if this is the BTC asset
            if name == 'BTCUSD' and latest_prediction_arima_1h != "N/A":
                last_time = df.index[-1]
                next_time = last_time + pd.Timedelta(hours=1)
                fig_main.add_trace(go.Scatter(
                    x=[last_time, next_time],
                    y=[latest_price, float(latest_prediction_arima_1h.replace(",", ""))],
                    mode='lines',
                    line=dict(color='#f7931a', width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig_main.add_trace(go.Scatter(
                    x=[next_time], 
                    y=[float(latest_prediction_arima_1h.replace(",", ""))], 
                    mode='markers', 
                    name='ARIMA 1H Forecast', 
                    marker=dict(size=10, color='#00ff00', symbol='diamond'),
                    hovertemplate='<b>AI Forecast:</b> $%{y:,.2f}<extra></extra>'
                ))

            fig_main.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0', family="Inter, sans-serif"),
                xaxis=dict(showgrid=False, showline=True, linecolor='#333', tickfont=dict(color='#666'), type='date'),
                yaxis=dict(showgrid=True, gridcolor='#1a1a1a', showline=False, tickfont=dict(color='#666'), side='right', tickformat='$,.2f'),
                margin=dict(l=0, r=0, t=10, b=0),
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color='#888'), bgcolor='rgba(0,0,0,0)'),
                hovermode='x unified'
            )
            main_plot = plot(fig_main, output_type='div', include_plotlyjs=False)
            
            asset_info.append({
                'name': name,
                'price': f"{latest_price:,.2f}",
                'change': f"{price_change:+,.2f}",
                'pct_change': f"{pct_change:+.2f}%",
                'is_positive': pct_change >= 0,
                'as_of': as_of,
                'arima_1h_prediction': arima_prediction_map.get(name, "N/A"),
                'signal': arima_snapshot_map.get(name, {}).get("signal", "N/A"),
                'signal_class': arima_snapshot_map.get(name, {}).get("signal_class", "secondary"),
                'call_value': f"{arima_snapshot_map.get(name, {}).get('last_close', 0):,.2f}" if arima_snapshot_map.get(name) else "N/A",
                'call_time': arima_snapshot_map.get(name, {}).get("run_time", "N/A"),
                'running_profit': "N/A",
                'running_profit_class': "secondary",
                'mini_plot': mini_plot,
                'main_plot': main_plot
            })

            if arima_snapshot_map.get(name):
                snap = arima_snapshot_map[name]
                running_pnl = (latest_price - snap["last_close"]) if snap["signal"] == "BUY" else (snap["last_close"] - latest_price)
                if running_pnl > 0:
                    pnl_class = "success"
                elif running_pnl < 0:
                    pnl_class = "danger"
                else:
                    pnl_class = "secondary"
                asset_info[-1]["running_profit"] = f"{running_pnl:+,.2f}"
                asset_info[-1]["running_profit_class"] = pnl_class

    tracked_assets = [a["name"] for a in asset_info]
    tracked_assets_text = ", ".join(tracked_assets) if tracked_assets else "BTCUSD, PAXUSD, SPX500, GOLD, NIFTY, USOIL"
    canonical_url = request.build_absolute_uri(request.path)
    site_name = "Intelligence.quantbots.co"
    seo_title = "Intelligence.quantbots.co | Machine Learning Market Intelligence for Finance"
    seo_description = (
        "Institutional-grade machine learning intelligence for financial markets. "
        f"Track live signals, ARIMA forecasts, and strategy performance across {tracked_assets_text}."
    )
    seo_keywords = (
        "machine learning finance, quantitative trading, algorithmic trading signals, "
        "ARIMA forecast, linear regression model, financial time series prediction, "
        f"{tracked_assets_text.lower()}, market intelligence dashboard"
    )
    current_iso = timezone.now().replace(microsecond=0).isoformat()

    organization_json_ld = {
        "@context": "https://schema.org",
        "@type": "Organization",
        "name": site_name,
        "url": request.build_absolute_uri("/"),
        "description": seo_description,
    }
    website_json_ld = {
        "@context": "https://schema.org",
        "@type": "WebSite",
        "name": site_name,
        "url": request.build_absolute_uri("/"),
        "description": seo_description,
        "inLanguage": "en-US",
    }
    webpage_json_ld = {
        "@context": "https://schema.org",
        "@type": "WebPage",
        "name": seo_title,
        "url": canonical_url,
        "description": seo_description,
        "dateModified": current_iso,
        "isPartOf": {"@type": "WebSite", "name": site_name, "url": request.build_absolute_uri("/")},
        "about": [
            "Machine Learning in Finance",
            "Quantitative Trading Signals",
            "ARIMA Forecasting",
            "Financial Time Series Prediction",
        ],
    }

    return render(request, 'homepage/landing.html', {
        'asset_info': asset_info,
        'latest_prediction_arima_1h': latest_prediction_arima_1h,
        'btc_latest_price': btc_latest_price,
        'btc_latest_change': btc_latest_change,
        'btc_latest_pct_change': btc_latest_pct_change,
        'btc_is_positive': btc_is_positive,
        'btc_as_of': btc_as_of,
        'seo_title': seo_title,
        'seo_description': seo_description,
        'seo_keywords': seo_keywords,
        'seo_canonical_url': canonical_url,
        'seo_site_name': site_name,
        'seo_locale': "en_US",
        'seo_updated_iso': current_iso,
        'seo_organization_json_ld': json.dumps(organization_json_ld),
        'seo_website_json_ld': json.dumps(website_json_ld),
        'seo_webpage_json_ld': json.dumps(webpage_json_ld),
    })

def asset_dashboard(request, asset_name):
    """
    Dedicated dashboard for a specific asset.
    """
    assets_data = get_landing_assets_data()
    df = assets_data.get(asset_name)

    if df is None or df.empty or len(df) < 2:
        return render(request, 'homepage/error.html', {'message': f'Insufficient data found for {asset_name}'})

    # Ensure index is datetime and sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Configure MLflow for prediction if applicable
    latest_prediction_lr = "N/A"
    if asset_name == 'BTCUSD':
        try:
            ml_dir = os.path.join(settings.BASE_DIR.parent, "ml")
            mlruns_path = os.path.join(ml_dir, "mlruns")
            tracking_uri = f"file://{mlruns_path}"
            mlflow.set_tracking_uri(tracking_uri)
            
            client = MlflowClient()
            exp = client.get_experiment_by_name("BTCUSD_Linear_Regression")
            if exp:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["attributes.start_time DESC"],
                    max_results=1
                )
                if runs:
                    pred = runs[0].data.metrics.get('predicted_next_hour_close', 'N/A')
                    if pred != "N/A":
                        latest_prediction_lr = f"{pred:.2f}"
        except Exception as e:
            print(f"Error fetching MLflow prediction for dashboard: {e}")

    # Ensure we have a single Series for 'Close'
    close_series = df['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]

    # Add technical indicators if needed (e.g., MA7)
    if len(df) >= 7:
        df['MA7'] = close_series.rolling(window=7).mean()

    # Latest price and change
    try:
        latest_price = float(close_series.iloc[-1])
        prev_price = float(close_series.iloc[-2])
        price_change = latest_price - prev_price
        pct_change = (price_change / prev_price) * 100
    except Exception as e:
        return render(request, 'homepage/error.html', {'message': f'Error processing data for {asset_name}: {str(e)}'})

    # Main interactive graph
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(
        x=df.index, 
        y=close_series, 
        mode='lines', 
        name=f'{asset_name} Price',
        line=dict(color='#f7931a', width=3),
        fill='tozeroy',
        fillcolor='rgba(247, 147, 26, 0.05)',
        hovertemplate='<b>Price:</b> $%{y:,.2f}<br><b>Date:</b> %{x}<extra></extra>'
    ))

    # Add technical indicators if needed (e.g., MA7)
    if 'MA7' in df.columns:
        fig_main.add_trace(go.Scatter(
            x=df.index, y=df['MA7'], mode='lines', name='MA7',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1)
        ))

    fig_main.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0', family="Inter, sans-serif"),
        xaxis=dict(showgrid=False, showline=True, linecolor='#333', tickfont=dict(color='#666'), type='date'),
        yaxis=dict(showgrid=True, gridcolor='#1a1a1a', showline=False, tickfont=dict(color='#666'), side='right', tickformat='$,.2f'),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color='#888'), bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified'
    )
    main_plot = plot(fig_main, output_type='div', include_plotlyjs=True)

    context = {
        'asset_name': asset_name,
        'latest_price': f"{latest_price:,.2f}",
        'price_change': f"{price_change:,.2f}",
        'pct_change': f"{pct_change:+.2f}%",
        'is_positive': pct_change >= 0,
        'main_plot': main_plot,
        'latest_prediction': latest_prediction_lr,
    }

    return render(request, 'homepage/asset_dashboard.html', context)
