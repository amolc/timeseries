import subprocess
import os
import logging
from celery import shared_task, chain
from celery.utils.log import get_task_logger
from celery.schedules import crontab

# Setup logging
logger = get_task_logger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Pipeline script paths
PIPELINES = {
    "BTCUSD": os.path.join(PROJECT_ROOT, "btcusd", "pipeline.py"),
    "GOLD": os.path.join(PROJECT_ROOT, "gold", "pipeline.py"),
    "NIFTY": os.path.join(PROJECT_ROOT, "nifty", "pipeline.py"),
    "PAXUSD": os.path.join(PROJECT_ROOT, "paxusd", "pipeline.py"),
    "SPX500": os.path.join(PROJECT_ROOT, "spx500", "pipeline.py")
}

@shared_task(name="tasks.run_asset_pipeline")
def run_asset_pipeline(asset, interval="1h"):
    """
    Executes the ML pipeline for a specific asset and interval.
    """
    script_path = PIPELINES.get(asset)
    if not script_path:
        logger.error(f"Pipeline script for {asset} not found.")
        return f"{asset}: Not Found"

    logger.info(f"Starting {interval} pipeline for {asset}...")
    try:
        # Standardized command for all assets: use --interval to run only one at a time.
        # This is more efficient for Celery chaining.
        cmd = ["python3", script_path, "--models", "lr,arima", "--interval", interval]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"{asset} {interval} pipeline executed successfully.")
        return f"{asset} {interval}: Success"
    except subprocess.CalledProcessError as e:
        logger.error(f"{asset} {interval} pipeline failed: {e}")
        logger.error(f"Error Output: {e.stderr}")
        return f"{asset} {interval}: Failed"
    except Exception as e:
        logger.error(f"Unexpected error in {asset} {interval} pipeline: {e}")
        return f"{asset} {interval}: Error"

@shared_task(name="tasks.run_interval_batch")
def run_interval_batch(interval):
    """
    Runs pipelines for all assets for a specific interval in a chain to avoid server overload.
    """
    logger.info(f"Starting batch run for {interval} interval...")
    
    # Create a chain of tasks to run sequentially
    # Using the .apply_async() or .delay() pattern for chains is preferred
    # but .s() and .apply_async() are the correct Celery way
    c = chain(
        run_asset_pipeline.si("BTCUSD", interval),
        run_asset_pipeline.si("GOLD", interval),
        run_asset_pipeline.si("NIFTY", interval),
        run_asset_pipeline.si("PAXUSD", interval),
        run_asset_pipeline.si("SPX500", interval)
    )
    return c.apply_async()

@shared_task(name="tasks.run_complete_cycle")
def run_complete_cycle():
    """
    Executes a complete cycle of all intervals for all assets in a proper sequential manner.
    """
    logger.info("Starting complete system-wide pipeline cycle...")
    
    # Chain all interval batches
    c = chain(
        run_interval_batch.si("1h"),
        run_interval_batch.si("1d"),
        run_interval_batch.si("1w"),
        run_interval_batch.si("1m")
    )
    return c.apply_async()

# Legacy task for backward compatibility if needed
@shared_task(name="tasks.run_ml_pipeline")
def run_ml_pipeline():
    return run_interval_batch("1h")
