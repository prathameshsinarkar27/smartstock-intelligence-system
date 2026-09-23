# register_scheduled_tasks.ps1
#
# Registers Windows Task Scheduler tasks for daily and weekly jobs.
# Provides a local alternative to the Docker scheduler.
#
# powershell -ExecutionPolicy Bypass -File scripts\windows\register_scheduled_tasks.ps1
#
# Parameters:
#   -DailyTime "18:00"        Daily job time (24h HH:mm).
#   -WeeklyDay "Sunday"       Weekly job day.
#   -WeeklyTime "19:00"       Weekly job time (24h HH:mm).
#   -RunWhenLoggedOff         Runs the task when the user is logged off.
#
# Example:
# powershell -ExecutionPolicy Bypass -File scripts\windows\register_scheduled_tasks.ps1 -DailyTime "06:30" -WeeklyDay "Saturday" -WeeklyTime "07:00"

param(
    [string]$DailyTime = "18:00",
    [string]$WeeklyDay = "Sunday",
    [string]$WeeklyTime = "19:00",
    [switch]$RunWhenLoggedOff
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DailyScript = Join-Path $ProjectRoot "scripts\windows\run_daily_job.ps1"
$WeeklyScript = Join-Path $ProjectRoot "scripts\windows\run_weekly_job.ps1"

if (-not (Test-Path $DailyScript) -or -not (Test-Path $WeeklyScript)) {
    Write-Error "Could not find run_daily_job.ps1 / run_weekly_job.ps1 under $ProjectRoot\scripts\windows\. Run this script from the project as-delivered."
    exit 1
}

function Register-SmartStockTask {
    param(
        [string]$TaskName,
        [string]$ScriptPath,
        [Microsoft.Management.Infrastructure.CimInstance]$Trigger
    )

    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Write-Output "Task '$TaskName' already exists — removing it first so this run's settings take effect."
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    $Action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-ExecutionPolicy Bypass -NoProfile -File `"$ScriptPath`""

    $Settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -WakeToRun `
        -DontStopOnIdleEnd `
        -ExecutionTimeLimit (New-TimeSpan -Hours 4)

    if ($RunWhenLoggedOff) {
        $Credential = Get-Credential -Message "Enter your Windows password so '$TaskName' can run when you're not logged in"
        Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
            -User $Credential.UserName -Password $Credential.GetNetworkCredential().Password `
            -RunLevel Highest -Description "SmartStock Intelligence Platform — Phase 14 automatic scheduler" | Out-Null
    } else {
        Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
            -Description "SmartStock Intelligence Platform — Phase 14 automatic scheduler" | Out-Null
    }

    Write-Output "Registered task '$TaskName'."
}

$DailyTrigger = New-ScheduledTaskTrigger -Daily -At $DailyTime
$WeeklyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $WeeklyDay -At $WeeklyTime

Register-SmartStockTask -TaskName "SmartStock Daily Job" -ScriptPath $DailyScript -Trigger $DailyTrigger
Register-SmartStockTask -TaskName "SmartStock Weekly Job" -ScriptPath $WeeklyScript -Trigger $WeeklyTrigger

Write-Output ""
Write-Output "Done. View/manage these tasks in Task Scheduler (taskschd.msc) under the Task Scheduler Library root,"
Write-Output "or run: Get-ScheduledTask -TaskName 'SmartStock *'"
Write-Output "To run one immediately (to test it without waiting for its scheduled time):"
Write-Output "  Start-ScheduledTask -TaskName 'SmartStock Daily Job'"
Write-Output "To remove both tasks: scripts\windows\unregister_scheduled_tasks.ps1"
