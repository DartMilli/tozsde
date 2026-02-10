param(
    [string]$Ticker = "VOO",
    [string]$StartDate = "2022-01-01",
    [string]$EndDate = "2023-12-31"
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $RepoRoot ".venv/Scripts/python.exe"

if (Test-Path $PythonPath) {
    $Python = $PythonPath
} else {
    $Python = "python"
}

& $Python "$RepoRoot/main.py" validate --ticker $Ticker --start-date $StartDate --end-date $EndDate
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $Python "$RepoRoot/scripts/phase6_check.py" --ticker $Ticker --start-date $StartDate --end-date $EndDate
exit $LASTEXITCODE
