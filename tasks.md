# Quantbots Intelligence Pipeline Tasks

This document explains the Celery-based scheduling and execution logic for the Quantbots Intelligence ML pipelines.

## Task Architecture

The system uses **Celery** with **Celery Beat** to manage recurring data ingestion and model training tasks. To avoid server overload, tasks are executed in **Chains**, ensuring that only one asset or interval is processed at a time.

### Core Tasks (`tasks.py`)

- **`run_asset_pipeline(asset, interval)`**: 
  The base unit of work. Executes the specific `pipeline.py` for a given asset (BTCUSD, GOLD, NIFTY, PAXUSD, SPX500) and interval (1h, 1d, 1w, 1m).
  
- **`run_interval_batch(interval)`**: 
  A **Chained Task** that runs all assets for a specific interval one after another.
  *Sequence: BTCUSD → GOLD → NIFTY → PAXUSD → SPX500*

- **`run_complete_cycle()`**: 
  A **Master Chain** that executes all intervals for all assets in a complete system-wide update.
  *Sequence: 1h Batch → 1d Batch → 1w Batch → 1m Batch*

---

## Scheduling Logic

The schedules are configured in `settings.py` using `CELERY_BEAT_SCHEDULE`.

| Frequency | Target Data | Schedule (UTC) | IST Equivalent |
| :--- | :--- | :--- | :--- |
| **Hourly** | 1 Hour Data | Every 60 minutes | Every hour |
| **Daily** | 1 Day Data | `00:00 UTC` | `05:30 AM IST` |
| **Weekly** | Complete Cycle | `Sunday 01:00 UTC` | `Sunday 06:30 AM IST` |

### Why Chaining?
Running multiple ML pipelines (which involve data fetching, technical indicator calculation, and model training like ARIMA/Linear Regression) simultaneously can consume significant CPU and Memory. 
**Chaining** ensures:
1. **Server Stability**: Prevents spikes in resource usage.
2. **Data Integrity**: Ensures one process completes before the next starts.
3. **Clean Logs**: Makes it easier to track the progress of the entire system update.

---

## Asset Pipelines
Each asset has its own dedicated directory and pipeline logic:
- `btcusd/pipeline.py` (BTCUSD)
- `gold/pipeline.py` (GOLD)
- `nifty/pipeline.py` (NIFTY)
- `paxusd/pipeline.py` (PAXUSD)
- `spx500/pipeline.py` (SPX500)
