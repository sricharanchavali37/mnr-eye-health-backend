@echo off
echo ================================================
echo   MNR Eye Health Platform - Backend
echo ================================================

cd /d "%~dp0"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo.
echo Starting FastAPI server on http://localhost:8000
echo API docs: http://localhost:8000/docs
echo Press Ctrl+C to stop
echo ================================================

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
