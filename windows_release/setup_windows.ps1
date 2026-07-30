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

Write-Host "[1/5] Checking Python 3.10+..." -ForegroundColor Cyan
$Python = Find-Python
if ($null -eq $Python) {
    if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        throw "Python 3.10+ was not found and winget is unavailable. Install 64-bit Python 3.11 from python.org, enable 'Add Python to PATH', then run this file again."
    }
    Write-Host "Python is missing. Installing Python 3.11 with winget..." -ForegroundColor Yellow
    & winget.exe install --exact --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget could not install Python 3.11."
    }
    $Python = Find-Python
    if ($null -eq $Python) {
        throw "Python was installed but is not visible yet. Close this window, reopen it, and run SETUP_WINDOWS.bat again."
    }
}

Write-Host "[2/5] Creating an isolated virtual environment..." -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    $PythonCommand = $Python.Command
    $PythonPrefix = $Python.Prefix
    & $PythonCommand $PythonPrefix -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create $VenvDir"
    }
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "[3/5] Installing Python packages (first run can take several minutes)..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip wheel
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $VenvPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "Package installation failed." }

Write-Host "[4/5] Verifying OpenCV, serial, AI model, and source files..." -ForegroundColor Cyan
& $VenvPython (Join-Path $ReleaseDir "verify_install.py")
if ($LASTEXITCODE -ne 0) { throw "Installation verification failed." }

Write-Host "[5/5] Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Before autonomous operation:" -ForegroundColor White
Write-Host "  1. Upload firmware\arm_controller\arm_controller.ino with Arduino IDE."
Write-Host "  2. Connect the charged external servo battery and common GND."
Write-Host "  3. Connect the Uno and wrist webcam."
Write-Host "  4. Double-click CHECK_CAMERA.bat once, then RUN_AUTONOMOUS.bat."
