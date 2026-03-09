import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from django.shortcuts import render
from django.conf import settings
import plotly.graph_objects as go
from plotly.offline import plot
from .utils import get_landing_assets_data

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
    latest_prediction_lr = "N/A"
    
    # Get LR Prediction for BTC (from 1h interval experiment in the new structure)
    try:
        # The new experiment name format is BTCUSD_LR_1h
        exp = client.get_experiment_by_name("BTCUSD_LR_1h")
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=1
            )
            if runs:
                # The metric name in the new pipeline is 'pred_next'
                pred = runs[0].data.metrics.get('pred_next', 'N/A')
                if pred != "N/A":
                    latest_prediction_lr = f"{pred:.2f}"
    except Exception as e:
        print(f"Error fetching MLflow prediction: {e}")

    assets_data = get_landing_assets_data()
    # Add PAXUSD to assets if needed, though get_landing_assets_data might already handle it
    # For now, let's ensure PAXUSD is in the list if we want it on the landing page
    if 'PAXUSD' not in assets_data:
        # This is just a safeguard, ideally get_landing_assets_data should be updated
        pass
    
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
            pct_change = (price_change / prev_price) * 100

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
            if name == 'BTCUSD' and latest_prediction_lr != "N/A":
                last_time = df.index[-1]
                next_time = last_time + pd.Timedelta(days=1)
                fig_main.add_trace(go.Scatter(
                    x=[last_time, next_time],
                    y=[latest_price, float(latest_prediction_lr)],
                    mode='lines',
                    line=dict(color='#f7931a', width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig_main.add_trace(go.Scatter(
                    x=[next_time], 
                    y=[float(latest_prediction_lr)], 
                    mode='markers', 
                    name='AI Forecast', 
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
                'change': f"{price_change:,.2f}",
                'pct_change': f"{pct_change:+.2f}%",
                'is_positive': pct_change >= 0,
                'mini_plot': mini_plot,
                'main_plot': main_plot
            })

    return render(request, 'homepage/landing.html', {
        'asset_info': asset_info,
        'latest_prediction': latest_prediction_lr
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

