@echo off
REM setup.bat — First-time setup: creates venv and installs all dependencies

echo.
echo  ====================================================
echo   RAG System Setup
echo  ====================================================
echo.

echo [1/3] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 ( echo ERROR: Python not found. & pause & exit /b 1 )

echo [2/3] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip

echo [3/3] Installing dependencies...
venv\Scripts\pip.exe install -r requirements.txt

echo.
echo  Setup complete! Run 'run.bat' to start the application.
pause
