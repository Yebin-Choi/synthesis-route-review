@echo off
rem Start the app so other computers on the same network can open it.
rem start.bat on its own stays on this machine only.
setlocal
set "SHARE=1"
call "%~dp0start.bat"
exit /b %errorlevel%
