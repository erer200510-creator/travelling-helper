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

rem --- Suppress Streamlit first-run email prompt ---
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
    echo [general]>"%USERPROFILE%\.streamlit\credentials.toml"
    echo email = "">>"%USERPROFILE%\.streamlit\credentials.toml"
)

echo ============================================
echo   AI Travel Agent - Web Chat
echo   URL: http://localhost:8501
echo   (this window must stay open; press Ctrl+C to stop)
echo ============================================
echo.

".venv\Scripts\python.exe" -m streamlit run app/ui/streamlit_app.py --server.port 8501 --browser.gatherUsageStats false

pause
