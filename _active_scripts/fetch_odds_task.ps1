# WC2026 — scheduled odds snapshot (closing-line fidelity for clv_tracker).
# Runs hourly via Windows Task Scheduler task "WC2026 odds snapshot".
# Appends DraftKings lines to data/raw/wc2026_match_odds.csv; the last
# snapshot before each kickoff becomes that match's closing line.
$repo = Split-Path $PSScriptRoot -Parent
$env:PYTHONUTF8 = "1"
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
Set-Location $repo
$log = Join-Path $repo "logs\odds_fetch.log"
"`n=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Add-Content $log
& "$repo\venv\Scripts\python.exe" -m src.models.fetch_live_odds 2>&1 |
    Select-Object -Last 5 | Add-Content $log
