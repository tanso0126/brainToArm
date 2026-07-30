param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Camera", "Diagnose", "Autonomous")]
    [string]$Tool,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ToolArgs
)

$ErrorActionPreference = "Stop"
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"

$ReleaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ReleaseDir
$PythonExe = Join-Path $RootDir ".venv-windows\Scripts\python.exe"

$Tools = @{
    Camera = @{
        Script = "windows_camera.py"
        Title = "손목 카메라 확인"
        DefaultArgs = @("--camera", "auto")
    }
    Diagnose = @{
        Script = "diagnose.py"
        Title = "카메라와 Arduino 연결 진단"
        DefaultArgs = @()
    }
    Autonomous = @{
        Script = "windows_app.py"
        Title = "자동 탐색, 접근, 잡기, HOME 복귀"
        DefaultArgs = @()
    }
}

$Definition = $Tools[$Tool]
$ScriptPath = Join-Path $ReleaseDir $Definition.Script

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " brainToArm - $($Definition.Title)"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Write-Host "[오류] Windows 실행환경이 아직 설치되지 않았습니다." -ForegroundColor Red
    Write-Host "먼저 windows_release\SETUP_WINDOWS.bat을 더블클릭하세요."
    Write-Host ""
    Read-Host "Enter 키를 누르면 창을 닫습니다"
    exit 2
}

if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
    Write-Host "[오류] 실행 파일을 찾을 수 없습니다:" -ForegroundColor Red
    Write-Host "  $ScriptPath"
    Write-Host "전체 배포 ZIP을 다시 내려받아 압축을 완전히 푸세요."
    Write-Host ""
    Read-Host "Enter 키를 누르면 창을 닫습니다"
    exit 2
}

if ($Tool -eq "Autonomous") {
    Write-Host "사람의 손과 헐거운 케이블을 로봇팔 이동 범위에서 치우세요." -ForegroundColor Yellow
    Write-Host "충전된 외부 서보 배터리와 공통 GND를 확인하세요." -ForegroundColor Yellow
    Write-Host ""
} elseif ($Tool -eq "Camera") {
    Write-Host "화면 하단에 파란색과 빨간색 집게가 모두 보여야 합니다."
    Write-Host "카메라 창에서 Q 또는 Esc를 누르면 종료됩니다."
    Write-Host ""
} else {
    Write-Host "이 진단은 로봇팔을 움직이지 않습니다."
    Write-Host ""
}

$Arguments = @("-u", $ScriptPath) + $Definition.DefaultArgs + $ToolArgs
& $PythonExe @Arguments
$Result = $LASTEXITCODE

Write-Host ""
if ($Result -eq 0) {
    Write-Host "[완료] 정상적으로 끝났습니다." -ForegroundColor Green
} else {
    Write-Host "[중단됨] 오류 코드 $Result 로 종료되었습니다." -ForegroundColor Red
    Write-Host "위쪽의 한국어 오류 내용을 확인하세요."
    if ($Tool -eq "Autonomous") {
        Write-Host "연결 문제는 DIAGNOSE.bat으로 확인할 수 있습니다."
    }
}
Write-Host ""
Read-Host "Enter 키를 누르면 창을 닫습니다"
exit $Result
