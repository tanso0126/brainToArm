@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_tool.ps1" -Tool Diagnose %*
exit /b %ERRORLEVEL%
