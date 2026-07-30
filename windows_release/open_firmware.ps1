$ErrorActionPreference = "Stop"

[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

$ReleaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ReleaseDir
$Candidates = @(
    (Join-Path $RootDir "firmware\arm_controller\arm_controller.ino"),
    (Join-Path $ReleaseDir "firmware\arm_controller\arm_controller.ino")
)
$Sketch = $Candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " brainToArm - Arduino 펌웨어 열기"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if ($null -eq $Sketch) {
    Write-Host "[오류] Arduino 펌웨어 파일을 찾을 수 없습니다." -ForegroundColor Red
    Write-Host ""
    Write-Host "확인한 위치:"
    foreach ($Candidate in $Candidates) {
        Write-Host "  $Candidate"
    }
    Write-Host ""
    Write-Host "windows_release 폴더만 따로 복사하거나 ZIP 안에서 바로 실행하면 안 됩니다."
    Write-Host "GitHub의 전체 Windows 배포 ZIP을 내려받아 압축을 완전히 푼 뒤,"
    Write-Host "압축을 푼 폴더 안의 windows_release\OPEN_FIRMWARE.bat을 실행하세요."
    Write-Host ""
    Read-Host "Enter 키를 누르면 창을 닫습니다"
    exit 2
}

$ArduinoExecutables = @()
if ($env:LOCALAPPDATA) {
    $ArduinoExecutables += Join-Path $env:LOCALAPPDATA `
        "Programs\Arduino IDE\Arduino IDE.exe"
}
if ($env:ProgramFiles) {
    $ArduinoExecutables += Join-Path $env:ProgramFiles `
        "Arduino IDE\Arduino IDE.exe"
}
if (${env:ProgramFiles(x86)}) {
    $ArduinoExecutables += Join-Path ${env:ProgramFiles(x86)} `
        "Arduino\arduino.exe"
}
$ArduinoExecutables = @(
    $ArduinoExecutables |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
)

Write-Host "[확인] 펌웨어 파일:" -ForegroundColor Green
Write-Host "  $Sketch"
Write-Host ""

try {
    if ($ArduinoExecutables.Count -gt 0) {
        Start-Process -FilePath $ArduinoExecutables[0] `
            -ArgumentList @("`"$Sketch`"")
    } else {
        Start-Process -FilePath $Sketch
    }
    Write-Host "[완료] Arduino IDE로 펌웨어를 열었습니다." -ForegroundColor Green
    Write-Host "보드에서 Arduino Uno와 올바른 COM 포트를 선택한 뒤 업로드하세요."
    Write-Host "업로드가 끝나면 Serial Monitor와 Serial Plotter를 닫으세요."
    $Result = 0
} catch {
    Write-Host "[오류] Arduino IDE로 파일을 열지 못했습니다." -ForegroundColor Red
    Write-Host $_.Exception.Message
    Write-Host ""
    Write-Host "Arduino IDE에서 다음 파일을 직접 여세요:"
    Write-Host "  $Sketch"
    Start-Process explorer.exe -ArgumentList @("/select,`"$Sketch`"")
    $Result = 3
}

Write-Host ""
Read-Host "Enter 키를 누르면 이 안내 창을 닫습니다"
exit $Result
