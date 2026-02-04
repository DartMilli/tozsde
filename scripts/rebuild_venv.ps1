# Rebuild Python virtual environment for the project.
# Usage: Run from repository root in PowerShell.

$root = "C:\tozsde"
$venv = Join-Path $root ".venv"
$py312 = "C:\Users\$env:USERNAME\AppData\Local\Microsoft\WindowsApps\python3.12.exe"

if (Test-Path $venv) {
    Remove-Item -Recurse -Force $venv
}

if (Test-Path $py312) {
    & $py312 -m venv $venv
} else {
    python -m venv $venv
}
& "$venv\Scripts\python.exe" -m pip install --upgrade pip
& "$venv\Scripts\python.exe" -m pip install -r "$root\requirements.txt"

Write-Host "Venv rebuilt. Activate with: $venv\Scripts\Activate.ps1"
