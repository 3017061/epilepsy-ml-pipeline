@echo off
REM Create venv (if missing), install requirements, and run smoke test
SETLOCAL
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Upgrading pip...
.venv\Scripts\python -m pip install --upgrade pip

echo Installing requirements...
.venv\Scripts\python -m pip install -r requirements.txt

echo Running smoke test...
.venv\Scripts\python scripts\smoke_test.py

ENDLOCAL
