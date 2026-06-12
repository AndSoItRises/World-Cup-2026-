# Registers the Windows Scheduled Task "WC2026 full refresh" that runs
# scripts/refresh_all.ps1 every 30 minutes (unattended).
#
# Idempotent: re-running re-registers cleanly. Run once from an ELEVATED or normal
# PowerShell:  .\scripts\register_refresh_task.ps1
# Remove after the final:  Unregister-ScheduledTask -TaskName "WC2026 full refresh" -Confirm:$false
#
# Note: this supersedes the hourly "WC2026 odds snapshot" task (refresh_all already
# fetches odds every cycle). Leaving both running is harmless — the old one just adds
# extra odds snapshots. Disable it if you want:
#   Disable-ScheduledTask -TaskName "WC2026 odds snapshot"

$ErrorActionPreference = "Stop"
$repo   = Split-Path $PSScriptRoot -Parent
$script = Join-Path $repo "scripts\refresh_all.ps1"
$name   = "WC2026 full refresh"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""

# Fire at the top of the hour, then every 30 min, indefinitely.
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date.AddHours((Get-Date).Hour) `
    -RepetitionInterval (New-TimeSpan -Minutes 30)

# Catch up after sleep/reboot; don't pile up if a cycle runs long.
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false
}

Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
    -Settings $settings -Description "WC2026 unattended pipeline: ingest results, reprice, rebuild dashboard, push when material." | Out-Null

Write-Host "✅ Registered '$name' — runs refresh_all.ps1 every 30 min (StartWhenAvailable)."
Write-Host "   Log: logs\refresh.log   |   Run now: Start-ScheduledTask -TaskName '$name'"
