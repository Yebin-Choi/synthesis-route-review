@echo off
pushd "%~dp0"
set "PY=%USERPROFILE%\miniconda3\envs\aizynth-env\python.exe"
if not exist "%PY%" goto NOPY
echo.
echo   Preloading demo targets. Takes about 10 min.
echo.
"%PY%" preload.py
popd
pause
goto END
:NOPY
echo Python env not found: %PY%
pause
:END
