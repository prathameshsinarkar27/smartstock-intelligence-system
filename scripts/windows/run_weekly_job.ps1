# run_weekly_job.ps1
#
# Runs the SmartStock weekly job (ML model retraining, evaluation, and a
# fresh prediction pass) once, then exits — meant to be registered as a
# Windows Task Scheduler task (see register_scheduled_tasks.ps1). See
# run_daily_job.ps1's header comment for the general pattern; this
# script mirrors it exactly, just for the weekly job.
#
# Can also be run manually for a one-off run:
#   powershell -ExecutionPolicy Bypass -File scripts\windows\run_weekly_job.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$LogFile = Join-Path $LogDir "scheduler-weekly-$Timestamp.log"

$VenvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvActivate)) {
    Write-Error "Virtual environment not found at $VenvActivate. Create it first: python -m venv venv"
    exit 1
}

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting weekly job. Logging to $LogFile" | Tee-Object -FilePath $LogFile -Append

& $VenvActivate
python -m src.scheduler.jobs weekly *>&1 | Tee-Object -FilePath $LogFile -Append
$ExitCode = $LASTEXITCODE

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Weekly job finished with exit code $ExitCode." | Tee-Object -FilePath $LogFile -Append

exit $ExitCode
