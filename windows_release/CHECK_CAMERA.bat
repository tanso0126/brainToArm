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
echo 카메라 창이 열립니다.
echo 화면 하단에 파란색과 빨간색 집게 테이프가 모두 보여야 합니다.
echo 카메라 창에서 Q 또는 Esc를 누르면 종료됩니다.
"%PYTHON_EXE%" -u "%CD%\windows_release\windows_camera.py" --camera auto %*
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
