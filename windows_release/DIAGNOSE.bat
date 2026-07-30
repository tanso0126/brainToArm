@echo off
setlocal
cd /d "%~dp0\.."
set "PYTHON_EXE=%CD%\.venv-windows\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Run windows_release\SETUP_WINDOWS.bat first.
  pause
  exit /b 2
)
"%PYTHON_EXE%" -u "%CD%\windows_release\diagnose.py" %*
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
