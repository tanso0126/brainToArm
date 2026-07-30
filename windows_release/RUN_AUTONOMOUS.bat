@echo off
setlocal
cd /d "%~dp0\.."
set "PYTHON_EXE=%CD%\.venv-windows\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Windows environment is missing.
  echo Double-click windows_release\SETUP_WINDOWS.bat first.
  pause
  exit /b 2
)
echo.
echo ============================================================
echo  brainToArm - autonomous find, approach, grasp, and HOME
echo ============================================================
echo Remove hands and loose cables from the robot workspace now.
echo Keep the external servo battery connected and charged.
echo.
"%PYTHON_EXE%" -u "%CD%\windows_release\windows_app.py" %*
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo [DONE] Autonomous run finished.
) else (
  echo [STOPPED] The run ended safely with error code %RESULT%.
  echo Read the message above. You can also run windows_release\DIAGNOSE.bat.
)
echo.
pause
exit /b %RESULT%
