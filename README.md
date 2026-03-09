# Market Intelligence Time Series Forecasting

A production-ready Market Intelligence system for forecasting major assets (BTC, GOLD, NIFTY, PAXG, SPX500) using Time Series models (Linear Regression, ARIMA). The project features a full MLOps lifecycle with experiment tracking, automated scheduling, and a comprehensive Django dashboard.

## **Core Architecture**

### **1. Forecasting Engine**
- **Models**: Linear Regression and ARIMA for multiple timeframes (1h, 1d, 1w, 1m).
- **Features**: Historical OHLCV data with technical indicators (RSI, Moving Averages, etc.).
- **Asset Coverage**: 
  - **BTCUSD**: Bitcoin / US Dollar
  - **GOLD**: Gold / US Dollar (TradingView: GOLD)
  - **NIFTY**: Nifty 50 Index (NSE: NIFTY)
  - **PAXUSD**: PAX Gold / USDT (Binance: PAXGUSDT)
  - **SPX500**: S&P 500 Index (OANDA: SPX500USD)

### **2. MLOps with MLflow**
- **Centralized Tracking**: All experiments are logged to a central SQLite database (`mlflow.db`).
- **Metric Tracking**: Logs MSE, MAE, and next-period price predictions (`pred_next`).
- **Artifact Management**: Stores trained models and diagnostic plots in `mlartifacts/`.

### **3. Automated Execution (Celery)**
- **Task Orchestration**: Uses Celery with Redis/RabbitMQ to manage complex task chains.
- **Sequential Execution**: Ensures assets are processed one-by-one to maintain server stability.
- **Dynamic Scheduling**:
  - **Hourly**: 1h data update for all assets.
  - **Daily**: 1d data update at 00:00 UTC.
  - **Weekly**: Full system cycle (1h, 1d, 1w, 1m) on Sundays.

### **4. Django Dashboard**
- **Real-time Monitoring**: Visualizes the latest predictions and model health.
- **ROI Analysis**: Calculates potential returns based on model signals.
- **Interactive Graphs**: Plotly-powered charts for historical data and forecasts.

---

## **Quick Start**

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start MLflow Tracking Server**:
   ```bash
   ./runmlflow.sh
   ```

3. **Start Celery Worker & Beat**:
   ```bash
   # Terminal 1: Worker (listening to the 'timeseries' queue)
   celery -A dashboard.timeseries_dashboard worker -Q timeseries --loglevel=info
   
   # Terminal 2: Beat
   celery -A dashboard.timeseries_dashboard beat --loglevel=info
   ```

4. **Start Django Dashboard**:
   ```bash
   cd dashboard
   python manage.py runserver 0.0.0.0:8000
   ```

---

## **Project Structure**

- `btcusd/`, `gold/`, `nifty/`, `paxusd/`, `spx500/`: Individual asset apps with data ingestion and modeling logic.
- `dashboard/`: Django project containing the monitoring and ROI modules.
- `tasks.py`: Celery task definitions and orchestration logic.
- `mlflow.db`: Centralized experiment database.
- `mlartifacts/`: Model storage.

---

## **Audit & Optimization Status**
- [x] Standardized pipeline structures across all assets.
- [x] Centralized MLflow configuration (SQLite).
- [x] Implemented sequential Celery task chaining.
- [x] Removed redundant legacy code and empty apps.
- [x] Updated documentation and execution scripts.

---
*This project is a critical lifeline for financial forecasting and is built with scalability and reliability as core principles.*

## **Proposed Project Structure**

```text
├── dashboard/              # Django Dashboard
│   ├── templates/          # HTML templates
│   ├── static/             # CSS/JS files
│   ├── apps/               # Monitoring, A/B Testing, ROI apps
│   └── manage.py           # Django entry point
├── src/                    # ML logic
│   ├── data/               # Data ingestion & preprocessing
│   ├── features/           # Feature engineering
│   ├── models/             # Model architectures & training
│   │   ├── linear_regression.py  # Linear Regression implementation
│   │   ├── arima_model.py        # ARIMA implementation
│   │   └── train.py              # RNN/General training script
│   ├── monitoring/         # Drift detection logic
│   └── tests/              # ML unit & integration tests
├── .github/workflows/      # CI/CD pipelines
├── data/                   # Raw and processed data
├── mlflow/                 # MLflow tracking server (if local)
├── config.yaml             # Project configuration
├── requirements.txt        # Dependencies
└── README.md               # Project plan and documentation
```

---

## **Getting Started**

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd timeseries
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run MLflow server**:
    ```bash
    mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri ./ml/mlruns --allowed-hosts "*" --cors-allowed-origins "*"
    ```
