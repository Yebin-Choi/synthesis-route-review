@echo off
setlocal

cd /d "%~dp0"
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" goto :directory_failed

if exist ".venv\" goto :existing_venv
py -3.11 -m venv ".venv"
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" goto :venv_failed
goto :venv_ready

:existing_venv
if exist ".venv\Scripts\python.exe" goto :venv_ready
echo [ERROR] The existing .venv is incomplete. It was not changed.
set "RESULT=1"
goto :failed

:venv_ready
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" goto :install_failed

".venv\Scripts\python.exe" -m streamlit run "app/app.py" --server.address localhost --browser.gatherUsageStats false
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" goto :app_failed
exit /b %RESULT%

:directory_failed
echo [ERROR] Could not switch to the project directory.
goto :failed

:venv_failed
echo [ERROR] Could not create the Python 3.11 virtual environment. Check that Python 3.11 and the py launcher are installed.
goto :failed

:install_failed
echo [ERROR] Dependency installation failed. The application was not started.
goto :failed

:app_failed
echo [ERROR] The application could not run. Check the error above.

:failed
echo.
pause
exit /b %RESULT%
