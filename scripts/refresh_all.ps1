# WC2026 — full unattended refresh orchestrator.
# Runs the entire pipeline end to end and pushes ONLY when something material
# changed, so the GitHub Pages dashboard stays live without commit spam.
#
# What it does each cycle:
#   1. fetch_live_odds      — snapshot current DraftKings lines        (always)
#   2. fetch_live_results   — auto-ingest finished matches from ESPN   (always)
#   3. IF a new/changed result landed:  live_update + predict_wc2026   (the ~3-min sims)
#   4. market_monitor, bet_sim, desk_call, clv_tracker                 (always — reprice)
#   5. build_dashboard + commit + push  — ONLY if a *meaningful* output changed
#      (a new result, or the desk verdicts / EV / model probs moved). Raw odds
#      jitter alone does NOT trigger a push — the dashboard's in-browser
#      "LIVE ODDS / auto-4m" refresh already keeps prices fresh client-side.
#
# Schedule it (every 30 min) with scripts/register_refresh_task.ps1, or run by hand.
# Log: logs/refresh.log

$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
$env:PYTHONUTF8 = "1"
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
Set-Location $repo
$py  = Join-Path $repo "venv\Scripts\python.exe"
$log = Join-Path $repo "logs\refresh.log"
function L($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Tee-Object -FilePath $log -Append }

L "════ refresh cycle start ════"

# 1. Odds snapshot (also captures closing lines for the CLV tracker)
& $py -m src.models.fetch_live_odds *>&1 | Select-Object -Last 3 | Add-Content $log

# 2. Results auto-ingest. Exit 10 = a result was added/changed, 0 = none, else = error.
& $py -m src.models.fetch_live_results *>&1 | Select-Object -Last 6 | Add-Content $log
$rc = $LASTEXITCODE
$newResult = ($rc -eq 10)
if ($rc -ne 0 -and $rc -ne 10) { L "WARN fetch_live_results exit=$rc (treating as no new result)" }
L "new result this cycle: $newResult"

# 3. Heavy sims ONLY when a real result landed
if ($newResult) {
    L "running live_update (ELO + 10k sims)…"
    & $py -m src.models.live_update    *>&1 | Select-Object -Last 4 | Add-Content $log
    L "running predict_wc2026 (calibrated match probs)…"
    & $py -m src.models.predict_wc2026 *>&1 | Select-Object -Last 3 | Add-Content $log
}

# 4. Reprice every cycle (cheap) so picks track the latest odds
& $py -m src.models.market_monitor *>&1 | Select-Object -Last 2 | Add-Content $log
& $py -m src.models.bet_sim        *>&1 | Select-Object -Last 2 | Add-Content $log
& $py -m src.models.desk_call      *>&1 | Select-Object -Last 2 | Add-Content $log
& $py -m src.models.clv_tracker    *>&1 | Select-Object -Last 2 | Add-Content $log

# 5. Push only on a meaningful change. These are the tracked outputs that, when they
#    move, mean the dashboard actually says something different. (line_movement.csv /
#    arb_scan.csv are deliberately excluded — they jitter with every odds tick.)
$trigger = @(
    "data/processed/desk_calls.csv",
    "data/processed/value_bets.csv",
    "data/processed/value_bets_futures.csv",
    "data/processed/wc2026_predictions.csv",
    "data/processed/tournament_probs_live.csv"
)
$dirty = git status --porcelain -- $trigger
if ($newResult -or $dirty) {
    L "material change → rebuilding dashboard + pushing"
    & $py -m src.models.build_dashboard *>&1 | Select-Object -Last 2 | Add-Content $log
    git add -A 2>&1 | Out-Null
    if (git status --porcelain) {
        $tag = if ($newResult) { "results+picks" } else { "picks" }
        $msg = "auto($tag): refresh $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
        git commit -m $msg 2>&1 | Add-Content $log
        git push 2>&1 | Add-Content $log
        if ($LASTEXITCODE -eq 0) { L "pushed ✅ ($msg)" } else { L "WARN push failed (will retry next cycle)" }
    } else { L "nothing staged after rebuild (no-op)" }
} else {
    L "no material change — skipped rebuild/push (odds jitter only)"
}

L "════ refresh cycle end ════"
