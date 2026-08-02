$ErrorActionPreference = "Stop"
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"

$ReleaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ReleaseDir
$PythonExe = Join-Path $RootDir ".venv-windows\Scripts\pythonw.exe"
$ConsolePython = Join-Path $RootDir ".venv-windows\Scripts\python.exe"
$Script = Join-Path $ReleaseDir "control_center.py"

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Write-Host "[오류] Windows 통합 실행환경이 설치되지 않았습니다." -ForegroundColor Red
    Write-Host "먼저 SETUP_WINDOWS.bat을 한 번 실행하세요."
    exit 2
}

try {
    Start-Process -FilePath $PythonExe -ArgumentList @($Script) -WorkingDirectory $RootDir
    Write-Host "brainToArm 통합 운영실을 여는 중입니다..." -ForegroundColor Cyan
    exit 0
} catch {
    Write-Host "[오류] 통합 운영실을 시작하지 못했습니다: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "자세한 원인을 보려면 아래 명령을 실행하세요:"
    Write-Host "  `"$ConsolePython`" `"$Script`""
    exit 2
}
