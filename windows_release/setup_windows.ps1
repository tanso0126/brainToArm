$ErrorActionPreference = "Stop"

[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"

$ReleaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ReleaseDir
$VenvDir = Join-Path $RootDir ".venv-windows"
$Requirements = Join-Path $ReleaseDir "requirements-windows.txt"
$DashboardDir = Join-Path $RootDir "dashboard"

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

Write-Host "[1/7] Python 3.10 이상이 설치되어 있는지 확인합니다..." -ForegroundColor Cyan
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

Write-Host "[2/7] GUI용 Node.js 22.13 이상을 확인합니다..." -ForegroundColor Cyan
$NodeReady = $false
if (Get-Command node.exe -ErrorAction SilentlyContinue) {
    & node.exe -e "const [a,b]=process.versions.node.split('.').map(Number);process.exit(a>22||(a===22&&b>=13)?0:1)"
    $NodeReady = ($LASTEXITCODE -eq 0)
}
if (-not $NodeReady) {
    if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        throw "Node.js 22.13 이상이 필요합니다. nodejs.org에서 LTS 64비트 버전을 설치한 뒤 다시 실행하세요."
    }
    Write-Host "GUI 실행에 필요한 Node.js LTS를 설치합니다..." -ForegroundColor Yellow
    & winget.exe install --exact --id OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "Node.js 설치에 실패했습니다. 인터넷 연결을 확인하세요." }
    $env:Path = "${env:ProgramFiles}\nodejs;${env:Path}"
    if (-not (Get-Command node.exe -ErrorAction SilentlyContinue)) {
        throw "Node.js는 설치되었지만 현재 창에서 아직 인식되지 않습니다. 창을 닫고 SETUP_WINDOWS.bat을 다시 실행하세요."
    }
}

Write-Host "[3/7] 이 프로젝트만 사용하는 독립 실행환경을 만듭니다..." -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    $PythonCommand = $Python.Command
    $PythonPrefix = $Python.Prefix
    & $PythonCommand $PythonPrefix -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "독립 실행환경을 만들지 못했습니다: $VenvDir"
    }
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "[4/7] EEG, 3D 시뮬레이션, 카메라 패키지를 설치합니다. 첫 설치는 수 분 이상 걸릴 수 있습니다..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip wheel
if ($LASTEXITCODE -ne 0) { throw "pip 업데이트에 실패했습니다. 인터넷 연결과 디스크 여유 공간을 확인하세요." }
& $VenvPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "필요한 패키지 설치에 실패했습니다. 인터넷 연결, 방화벽, 디스크 여유 공간을 확인한 뒤 다시 실행하세요." }

Write-Host "[5/7] 전용 GUI 구성 요소를 설치하고 빌드합니다..." -ForegroundColor Cyan
Push-Location $DashboardDir
try {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw "GUI 패키지 설치에 실패했습니다. 인터넷 연결과 방화벽을 확인하세요." }
    & npm.cmd run build:windows
    if ($LASTEXITCODE -ne 0) { throw "Windows 내장 GUI 빌드에 실패했습니다. 저장소를 다시 내려받으세요." }
} finally {
    Pop-Location
}

Write-Host "[6/7] EEG, 3D 엔진, 카메라, 시리얼 통신, AI 모델을 검사합니다..." -ForegroundColor Cyan
& $VenvPython (Join-Path $ReleaseDir "verify_install.py")
if ($LASTEXITCODE -ne 0) { throw "설치 결과 검사에 실패했습니다. 바로 위에 표시된 오류를 확인하세요." }

Write-Host "[7/7] 설치가 완료되었습니다." -ForegroundColor Green
Write-Host ""
Write-Host "자동 실행 전에 할 일:" -ForegroundColor White
Write-Host "  1. Arduino IDE로 firmware\arm_controller\arm_controller.ino를 업로드합니다."
Write-Host "  2. 충전된 외부 서보 배터리와 공통 GND를 연결합니다."
Write-Host "  3. Uno USB와 손목 웹캠을 연결합니다."
Write-Host "  4. 이후에는 START_CONTROL_CENTER.bat 하나만 실행합니다."
