$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path "$root\.venv\Scripts\python.exe")) { Write-Host "Create the venv first: py -3.14 -m venv .venv; .venv\Scripts\pip install -r requirements.txt"; exit 1 }
$proc = Start-Process -PassThru -NoNewWindow "$root\.venv\Scripts\python.exe" -ArgumentList @(
    "-m", "uvicorn", "api.main:app", "--app-dir", "$root\backend", "--host", "127.0.0.1", "--port", "8000"
)
$ready = $false
for ($i = 0; $i -lt 150; $i++) {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:8000/" -UseBasicParsing -TimeoutSec 1 | Out-Null
        $ready = $true
        break
    } catch {
        Start-Sleep -Milliseconds 200
    }
}
if (-not $ready) { Write-Host "Server did not answer within 30 s; check the uvicorn output." }
Start-Process "http://127.0.0.1:8000"
Wait-Process -Id $proc.Id
