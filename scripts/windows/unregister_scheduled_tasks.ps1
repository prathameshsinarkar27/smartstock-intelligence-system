# unregister_scheduled_tasks.ps1
#
# Removes the two Windows Task Scheduler tasks created by
# register_scheduled_tasks.ps1. Safe to run even if they don't exist.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\windows\unregister_scheduled_tasks.ps1

$ErrorActionPreference = "Stop"

foreach ($TaskName in @("SmartStock Daily Job", "SmartStock Weekly Job")) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Output "Removed task '$TaskName'."
    } else {
        Write-Output "Task '$TaskName' was not registered — nothing to remove."
    }
}
