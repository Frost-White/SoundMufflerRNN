$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$reportsDir = Join-Path $root "reports"
New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null

$backendExit = 0
$frontendExit = 0

Write-Host "=== Backend tests (pytest) ==="
Push-Location $root
python -m pytest -q tests/backend --junitxml "$reportsDir/backend-junit.xml"
$backendExit = $LASTEXITCODE
Pop-Location

Write-Host "`n=== Frontend tests (vitest) ==="
Push-Location (Join-Path $root "frontend/app")
npm run test:report
$frontendExit = $LASTEXITCODE
Pop-Location

Write-Host "`n=== Test report summary ==="
Write-Host "backend exit code : $backendExit"
Write-Host "frontend exit code: $frontendExit"
Write-Host "reports folder    : $reportsDir"

if ($backendExit -ne 0 -or $frontendExit -ne 0) {
  Write-Host "status            : FAIL (check junit xml files)"
  exit 1
}

Write-Host "status            : PASS"
exit 0
