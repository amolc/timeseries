#!/bin/bash

# Get the absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Navigate to the dashboard directory where manage.py is located
cd "$PROJECT_ROOT/dashboard"

echo "Starting Django Development Server..."
# Run the Django server
python3 manage.py runserver 9900
