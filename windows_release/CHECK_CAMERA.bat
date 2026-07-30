@echo off
setlocal
cd /d "%~dp0\.."
set "PYTHON_EXE=%CD%\.venv-windows\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Run windows_release\SETUP_WINDOWS.bat first.
  pause
  exit /b 2
)
echo A camera window will open. Both blue and red finger tapes must be visible.
echo Press Q in the camera window to close it.
"%PYTHON_EXE%" -u "%CD%\windows_release\windows_camera.py" --camera auto %*
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
