$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path "$root\.venv\Scripts\python.exe")) { Write-Host "Create the venv first: py -3.14 -m venv .venv; .venv\Scripts\pip install -r requirements.txt"; exit 1 }
Start-Process "http://127.0.0.1:8000"
& "$root\.venv\Scripts\python.exe" -m uvicorn api.main:app --app-dir "$root\backend" --host 127.0.0.1 --port 8000
