@echo off
setlocal
pushd "%~dp0"
set "PY=%USERPROFILE%\miniconda3\envs\aizynth-env\python.exe"
if not exist "%PY%" (
    echo Python environment not found: %PY%
    popd
    exit /b 1
)
echo Running OFFLINE example analysis only. No live model or API cache is prepared.
"%PY%" preload.py
set "RESULT=%ERRORLEVEL%"
popd
pause
exit /b %RESULT%
