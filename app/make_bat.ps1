$dir = Split-Path -Parent $MyInvocation.MyCommand.Path

$run = @(
  '@echo off'
  'pushd "%~dp0"'
  'set "PY=%USERPROFILE%\miniconda3\envs\aizynth-env\python.exe"'
  'set "PORT=8501"'
  ''
  'if not exist "%PY%" goto NOPY'
  ''
  'rem already running? just open the browser'
  'netstat -ano | findstr /r /c:"LISTENING" | findstr ":%PORT% " >nul 2>&1'
  'if %errorlevel%==0 goto ALREADY'
  ''
  'echo.'
  'echo   Starting. Browser opens in 2-3 min (loading 754MB model).'
  'echo   Close this window to stop.'
  'echo.'
  'start "" http://localhost:%PORT%'
  '"%PY%" -m streamlit run app.py --server.port %PORT%'
  'popd'
  'pause'
  'goto END'
  ''
  ':ALREADY'
  'echo.'
  'echo   Already running. Opening browser.'
  'echo.'
  'start "" http://localhost:%PORT%'
  'popd'
  'timeout /t 3 >nul'
  'goto END'
  ''
  ':NOPY'
  'echo Python env not found: %PY%'
  'echo Run: conda create -n aizynth-env python=3.11 -y'
  'pause'
  ':END'
)
[System.IO.File]::WriteAllLines("$dir\run.bat", $run, [System.Text.Encoding]::ASCII)

$stop = @(
  '@echo off'
  'echo Stopping app on port 8501...'
  'for /f "tokens=5" %%a in (''netstat -ano ^| findstr ":8501 " ^| findstr LISTENING'') do taskkill /f /pid %%a >nul 2>&1'
  'echo Done.'
  'timeout /t 2 >nul'
)
[System.IO.File]::WriteAllLines("$dir\stop.bat", $stop, [System.Text.Encoding]::ASCII)

$pre = @(
  '@echo off'
  'pushd "%~dp0"'
  'set "PY=%USERPROFILE%\miniconda3\envs\aizynth-env\python.exe"'
  'if not exist "%PY%" goto NOPY'
  'echo.'
  'echo   Preloading demo targets. Takes about 10 min.'
  'echo.'
  '"%PY%" preload.py'
  'popd'
  'pause'
  'goto END'
  ':NOPY'
  'echo Python env not found: %PY%'
  'pause'
  ':END'
)
[System.IO.File]::WriteAllLines("$dir\preload.bat", $pre, [System.Text.Encoding]::ASCII)

Write-Output "run.bat / stop.bat / preload.bat 생성 완료"
