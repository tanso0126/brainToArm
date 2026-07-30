@echo off
setlocal
cd /d "%~dp0"
echo.
echo ============================================================
echo  brainToArm - Windows first-time setup
echo ============================================================
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" (
  echo [FAILED] Setup stopped with error code %RESULT%.
  echo Read the red error above or run DIAGNOSE.bat after fixing it.
) else (
  echo [OK] Setup finished. Next, double-click RUN_AUTONOMOUS.bat.
)
echo.
pause
exit /b %RESULT%
