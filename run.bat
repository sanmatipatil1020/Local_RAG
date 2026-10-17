@echo off
REM ═══════════════════════════════════════════════════════════════════
REM  run.bat — Privacy-Preserving Local RAG System
REM  Run this file to start the application.
REM ═══════════════════════════════════════════════════════════════════

echo.
echo  ====================================================
echo   Privacy-Preserving Local RAG System
echo  ====================================================
echo.

REM Step 1: Check Ollama is running
echo [1/3] Checking Ollama...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Ollama is not running or not installed.
    echo  Please start Ollama first: open a new terminal and run 'ollama serve'
    echo  Then run this script again.
    pause
    exit /b 1
)
echo  OK - Ollama is running.

REM Step 2: Activate virtual environment
echo [2/3] Activating virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo  ERROR: Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

REM Step 3: Start Flask
echo [3/3] Starting Flask server...
echo.
echo  Open your browser at: http://localhost:5000
echo  Press Ctrl+C to stop the server.
echo.
python app.py

pause
