#!/bin/bash

# Get the absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Add both root and dashboard to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/dashboard"

# Navigate to the dashboard directory where celery config is located
cd "$PROJECT_ROOT/dashboard"

echo "Starting Celery Worker..."
# Run the worker process
celery -A timeseries_dashboard worker --loglevel=info -Q timeseries -I tasks -n timeseries-worker@%h
