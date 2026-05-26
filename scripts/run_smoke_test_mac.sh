#!/usr/bin/env bash
set -euo pipefail

echo "Setting up Python virtual environment at .venv (macOS)..."

# Choose python executable
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "No Python executable found. Install Python 3 and retry."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating virtual environment using $PY..."
    $PY -m venv .venv
fi

echo "Upgrading pip..."
.venv/bin/python -m pip install --upgrade pip

echo "Installing requirements from requirements.txt..."
if [ -f "requirements.txt" ]; then
    .venv/bin/python -m pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping dependency install."
fi

echo "Running smoke test script..."
.venv/bin/python scripts/smoke_test.py

echo "Smoke test finished."
