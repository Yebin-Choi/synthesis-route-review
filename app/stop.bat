@echo off
echo Stopping app on port 8501...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501 " ^| findstr LISTENING') do taskkill /f /pid %%a >nul 2>&1
echo Done.
timeout /t 2 >nul
