@echo off
echo Starting SABDE - State-of-the-Art Biomedical Discovery Engine
echo ==============================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is required but not installed.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is required but not installed.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Install Node.js dependencies
echo Installing Node.js dependencies...
cd frontend
call npm install
cd ..

REM Initialize data
echo Initializing sample data...
python scripts/init_data.py

REM Start services
echo Starting services...

REM Start FastAPI backend
echo Starting FastAPI backend on http://localhost:8000
start "SABDE Backend" cmd /k "uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start React frontend
echo Starting React frontend on http://localhost:3000
cd frontend
start "SABDE Frontend" cmd /k "npm start"
cd ..

echo SABDE is now running!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to stop all services
pause >nul
