$ErrorActionPreference = "Stop"

$ReleaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ReleaseDir
$VenvDir = Join-Path $RootDir ".venv-windows"
$Requirements = Join-Path $ReleaseDir "requirements-windows.txt"

function Find-Python {
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        try {
            & py.exe -3.11 -c "import sys; assert sys.version_info >= (3, 10)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @{ Command = "py.exe"; Prefix = @("-3.11") }
            }
        } catch {}
    }
    if (Get-Command python.exe -ErrorAction SilentlyContinue) {
        try {
            & python.exe -c "import sys; assert sys.version_info >= (3, 10)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @{ Command = "python.exe"; Prefix = @() }
            }
        } catch {}
    }
    return $null
}

Write-Host "[1/5] Python 3.10 이상이 설치되어 있는지 확인합니다..." -ForegroundColor Cyan
$Python = Find-Python
if ($null -eq $Python) {
    if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        throw "Python 3.10 이상과 winget을 모두 찾지 못했습니다. python.org에서 64비트 Python 3.11을 설치하고 'Add Python to PATH'를 선택한 뒤 이 파일을 다시 실행하세요."
    }
    Write-Host "Python이 없습니다. winget으로 Python 3.11을 설치합니다..." -ForegroundColor Yellow
    & winget.exe install --exact --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget으로 Python 3.11을 설치하지 못했습니다. 인터넷 연결을 확인하거나 python.org에서 직접 설치하세요."
    }
    $Python = Find-Python
    if ($null -eq $Python) {
        throw "Python은 설치되었지만 현재 창에서 아직 인식되지 않습니다. 이 창을 닫고 SETUP_WINDOWS.bat을 다시 실행하세요."
    }
}

Write-Host "[2/5] 이 프로젝트만 사용하는 독립 실행환경을 만듭니다..." -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    $PythonCommand = $Python.Command
    $PythonPrefix = $Python.Prefix
    & $PythonCommand $PythonPrefix -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "독립 실행환경을 만들지 못했습니다: $VenvDir"
    }
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "[3/5] 필요한 Python 패키지를 설치합니다. 첫 설치는 수 분 이상 걸릴 수 있습니다..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip wheel
if ($LASTEXITCODE -ne 0) { throw "pip 업데이트에 실패했습니다. 인터넷 연결과 디스크 여유 공간을 확인하세요." }
& $VenvPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "필요한 패키지 설치에 실패했습니다. 인터넷 연결, 방화벽, 디스크 여유 공간을 확인한 뒤 다시 실행하세요." }

Write-Host "[4/5] 카메라, 시리얼 통신, AI 모델, 소스 파일을 검사합니다..." -ForegroundColor Cyan
& $VenvPython (Join-Path $ReleaseDir "verify_install.py")
if ($LASTEXITCODE -ne 0) { throw "설치 결과 검사에 실패했습니다. 바로 위에 표시된 오류를 확인하세요." }

Write-Host "[5/5] 설치가 완료되었습니다." -ForegroundColor Green
Write-Host ""
Write-Host "자동 실행 전에 할 일:" -ForegroundColor White
Write-Host "  1. Arduino IDE로 firmware\arm_controller\arm_controller.ino를 업로드합니다."
Write-Host "  2. 충전된 외부 서보 배터리와 공통 GND를 연결합니다."
Write-Host "  3. Uno USB와 손목 웹캠을 연결합니다."
Write-Host "  4. CHECK_CAMERA.bat으로 확인한 뒤 RUN_AUTONOMOUS.bat을 실행합니다."
