@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0open_firmware.ps1"
exit /b %ERRORLEVEL%
