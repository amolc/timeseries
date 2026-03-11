from __future__ import annotations

import logging
import os
from typing import Optional


logger = logging.getLogger(__name__)


def notify_mlflow_run_summary(
    asset_name: str,
    model_name: str,
    interval: str,
    *,
    status: str = "completed",
    details: Optional[str] = None,
) -> bool:
    """Best-effort notification hook used by training pipelines.

    This helper must never raise, so training jobs remain reliable even when
    external notification systems are not configured.
    """
    try:
        webhook = os.environ.get("MLFLOW_NOTIFY_WEBHOOK", "").strip()
        message = f"{asset_name} {model_name} {interval} run {status}"
        if details:
            message = f"{message}: {details}"

        if not webhook:
            logger.info("MLflow notify: %s", message)
            return False

        # Webhook delivery is optional. Keep import local and guarded.
        import requests  # type: ignore

        response = requests.post(webhook, json={"text": message}, timeout=5)
        if response.ok:
            return True

        logger.warning(
            "MLflow notify webhook failed with status %s for %s",
            response.status_code,
            message,
        )
        return False
    except Exception as exc:  # pragma: no cover
        logger.warning("MLflow notify failed: %s", exc)
        return False
