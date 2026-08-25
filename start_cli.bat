@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found. Please run:
    echo   python -m venv .venv
    echo   .venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo ============================================
echo   AI Travel Agent - CLI Chat
echo   Type exit to quit
echo ============================================
echo.

".venv\Scripts\python.exe" -m app.ui.cli

pause
