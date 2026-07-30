@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0\.."
set "PYTHON_EXE=%CD%\.venv-windows\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [오류] 먼저 windows_release\SETUP_WINDOWS.bat을 실행하세요.
  pause
  exit /b 2
)
echo 로봇팔을 움직이지 않고 카메라와 COM 포트만 확인합니다.
"%PYTHON_EXE%" -u "%CD%\windows_release\diagnose.py" %*
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
