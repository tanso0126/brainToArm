@echo off
setlocal
set "SKETCH=%~dp0..\firmware\arm_controller\arm_controller.ino"
if not exist "%SKETCH%" (
  echo [ERROR] Firmware sketch not found: %SKETCH%
  pause
  exit /b 2
)
start "" "%SKETCH%"
