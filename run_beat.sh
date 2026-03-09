#!/bin/bash

# Get the absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Add both root and dashboard to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/dashboard"

# Navigate to the dashboard directory where celery config is located
cd "$PROJECT_ROOT/dashboard"

echo "Starting Celery Beat..."
# Run the beat scheduler process
celery -A timeseries_dashboard beat --loglevel=info --schedule /tmp/timeseries-celerybeat-schedule
