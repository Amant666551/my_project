$ErrorActionPreference = "Stop"

$desktopDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $desktopDir
Set-Location $projectRoot

$venvDir = Join-Path $projectRoot ".venv310"
$venvCfg = Join-Path $venvDir "pyvenv.cfg"
$sitePackages = Join-Path $venvDir "Lib\site-packages"

if (-not (Test-Path $venvCfg)) {
    throw "pyvenv.cfg not found: $venvCfg"
}

$pythonHome = $null
foreach ($line in Get-Content $venvCfg) {
    if ($line -match '^\s*home\s*=\s*(.+?)\s*$') {
        $pythonHome = $matches[1].Trim()
        break
    }
}

if (-not $pythonHome) {
    throw "Unable to read Python home from $venvCfg"
}

$pythonExe = Join-Path $pythonHome "python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Base python not found: $pythonExe"
}

if (-not (Test-Path $sitePackages)) {
    throw "site-packages not found: $sitePackages"
}

$env:PYTHONPATH = $sitePackages

& $pythonExe -c "import webview" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "pywebview not found in build environment. Installing into .venv site-packages..."
    & $pythonExe -m pip install --upgrade --target $sitePackages pywebview
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install pywebview into $sitePackages"
    }
}

& $pythonExe -m PyInstaller --clean --noconfirm (Join-Path $desktopDir "desktop_app.spec")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Build complete."
Write-Host "Output: $projectRoot\dist\SpeechTranslator.exe"
