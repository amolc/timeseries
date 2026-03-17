import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

CHANNEL_MAP = {
    "BTCUSD": "-1003814812761",
    "PAXUSD": "-1003777616319",
    "GOLD": "-1003812463342",
    "SPX500": "-1003687675372",
    "NIFTY": "-1003827167426",
    "USOIL": "-1003759889690",
}

def send_switchover_alert(asset: str, new_signal: str) -> bool:
    """Best-effort notification hook used by training pipelines.

    This helper must never raise, so training jobs remain reliable even when
    external notification systems are not configured.
    """
    try:
        # Load the token. We fallback to the user's provided token directly to ensure it works.
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "8596334331:AAHaQJmXKi1xm2z_wSiM-O2OdKofzB_COy0").strip()
        
        chat_id = CHANNEL_MAP.get(asset.upper())
        if not chat_id:
            logger.warning("Telegram notify: No channel mapped for asset %s", asset)
            return False

        if not token:
            logger.info("Telegram notify skipped: TELEGRAM_BOT_TOKEN not set.")
            return False

        icon = "🟢" if new_signal.upper() == "BUY" else "🔴"
        message = f"🚨 *SWITCHOVER ALERT: {asset.upper()}*\n\nThe model sequence has flipped to a new *{icon} {new_signal.upper()}* signal."
        url = f"https://api.telegram.org/bot{token}/sendMessage"

        # Webhook delivery is optional. Keep import local and guarded.
        import requests  # type: ignore

        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        response = requests.post(url, json=payload, timeout=5)
        if response.ok and response.json().get("ok"):
            logger.info("Telegram notification sent for %s: %s", asset, new_signal)
            return True

        logger.warning(
            "Telegram notify failed with status %s for %s. Response: %s",
            response.status_code,
            asset,
            response.text
        )
        return False
    except Exception as exc:  # pragma: no cover
        logger.warning("Telegram notify failed: %s", exc)
        return False

def check_and_send_switchovers(asset: str, interval: str):
    """
    Queries MLflow for the latest two runs of LR and/or ARIMA for the given asset/interval.
    Calculates if a switchover occurred between the previous run and the current run.
    Sends a Telegram alert if a switchover is detected.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        # MLflow setup
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mlflow_db_path = os.path.join(project_root, "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
        client = MlflowClient()

        models_to_check = {
            "LR": f"{asset.upper()}_LR_{interval}",
            "ARIMA": f"{asset.upper()}_ARIMA_{interval}"
        }

        for model_key, exp_name in models_to_check.items():
            exp = client.get_experiment_by_name(exp_name)
            if not exp:
                continue

            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=2
            )

            if len(runs) >= 2:
                curr_run = runs[0]
                prev_run = runs[1]

                def get_signal(run):
                    pred = run.data.metrics.get('pred_next')
                    close = run.data.metrics.get('last_close_price')
                    if pred is None or close is None:
                        return None
                    return "BUY" if pred > close else "SELL"

                curr_signal = get_signal(curr_run)
                prev_signal = get_signal(prev_run)

                if curr_signal and prev_signal and curr_signal != prev_signal:
                    logger.info(f"Switchover detected for {asset} {model_key} {interval}: {prev_signal} -> {curr_signal}")
                    send_switchover_alert(asset, f"{curr_signal} ({model_key})")

    except Exception as e:
        logger.warning(f"Error checking switchovers for Telegram: {e}")
