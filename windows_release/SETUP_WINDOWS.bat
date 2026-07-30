@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
