@echo off
title Synthesis Route Impurity Review
pushd "%~dp0" || exit /b 1
set "PY=%USERPROFILE%\miniconda3\envs\aizynth-env\python.exe"

if not exist "%PY%" goto NOPY

echo.
echo   Starting the research screening interface at http://localhost:8501
echo   Offline examples do not load route-generation models or call external APIs.
echo   Route generation and public ASKCOS requests run only when selected.
echo   Close this window or press Ctrl+C to stop this app.
echo.
"%PY%" -m streamlit run "%~dp0app.py" --server.address localhost --server.port 8501 --browser.gatherUsageStats false
set "EXIT_CODE=%errorlevel%"
popd
pause
exit /b %EXIT_CODE%

:NOPY
echo Python environment not found: %PY%
echo Install the project dependencies in the aizynth-env environment before launching.
popd
pause
exit /b 1
