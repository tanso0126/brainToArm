@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0\.."
set "PYTHON_EXE=%CD%\.venv-windows\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [오류] Windows 실행환경이 설치되어 있지 않습니다.
  echo 먼저 windows_release\SETUP_WINDOWS.bat을 더블클릭하세요.
  pause
  exit /b 2
)
echo.
echo ============================================================
echo  brainToArm - 자동 탐색, 접근, 잡기, HOME 복귀
echo ============================================================
echo 지금 사람의 손과 헐거운 케이블을 로봇팔 이동 범위에서 치우세요.
echo 충전된 외부 서보 배터리가 연결되어 있어야 합니다.
echo.
"%PYTHON_EXE%" -u "%CD%\windows_release\windows_app.py" %*
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo [완료] 자동 실행이 정상적으로 끝났습니다.
) else (
  echo [중단됨] 오류 코드 %RESULT%^(으^)로 실행이 종료되었습니다.
  echo 위쪽의 한국어 오류 내용을 읽어보세요.
  echo 연결 문제는 windows_release\DIAGNOSE.bat으로 확인할 수 있습니다.
)
echo.
pause
exit /b %RESULT%
