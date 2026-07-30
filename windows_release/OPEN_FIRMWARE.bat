@echo off
chcp 65001 >nul
setlocal
set "SKETCH=%~dp0..\firmware\arm_controller\arm_controller.ino"
if not exist "%SKETCH%" (
  echo [오류] Arduino 펌웨어 파일을 찾을 수 없습니다.
  echo 찾으려던 위치: %SKETCH%
  echo 저장소의 압축을 완전히 풀었는지 확인하세요.
  pause
  exit /b 2
)
start "" "%SKETCH%"
