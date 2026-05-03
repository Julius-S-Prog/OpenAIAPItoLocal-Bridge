@echo off
setlocal enabledelayedexpansion

echo ========================================
echo  OpenAI to llama.cpp Bridge
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Installing/upgrading dependencies...
pip install -r requirements.txt --upgrade
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

REM Create .env from .env.example if .env does not exist
if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env from .env.example...
        copy .env.example .env >nul
        echo.
        echo Please edit .env if you need to change the llama.cpp server address.
        echo.
    )
)

echo.
echo Starting bridge on http://localhost:8000
echo Press Ctrl+C to stop.
echo.

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
