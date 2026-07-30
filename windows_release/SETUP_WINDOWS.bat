@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo.
echo ============================================================
echo  brainToArm - Windows 최초 설치
echo ============================================================
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" (
  echo [설치 실패] 오류 코드: %RESULT%
  echo 위쪽의 빨간색 오류 내용을 읽고 문제를 해결한 뒤 다시 실행하세요.
  echo 하드웨어 연결 상태는 DIAGNOSE.bat으로 확인할 수 있습니다.
) else (
  echo [설치 완료] 다음에는 CHECK_CAMERA.bat으로 카메라를 확인하세요.
  echo 모든 준비가 끝나면 RUN_AUTONOMOUS.bat을 더블클릭하세요.
)
echo.
pause
exit /b %RESULT%
