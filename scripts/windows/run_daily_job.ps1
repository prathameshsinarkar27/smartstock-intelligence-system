# run_daily_job.ps1
#
# Runs the SmartStock daily job (ingestion + ETL, sentiment scoring,
# ML predictions with the current model) once, then exits — meant to be
# registered as a Windows Task Scheduler task (see
# register_scheduled_tasks.ps1), which handles the "run this every day
# at HH:MM" recurrence natively. This script itself does not loop or
# wait; Task Scheduler re-launches it on schedule.
#
# Can also be run manually:
#   powershell -ExecutionPolicy Bypass -File scripts\windows\run_daily_job.ps1
#
# Exit code is forwarded from `python -m src.scheduler.jobs daily`
# (0 = every step succeeded, 1 = at least one step failed), so Task
# Scheduler's "Last Run Result" column reflects real job outcomes.

$ErrorActionPreference = "Stop"

# Resolve the project root as two levels up from this script's location
# (scripts\windows\run_daily_job.ps1 -> project root), so this works
# regardless of the working directory Task Scheduler launches it from.
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$LogFile = Join-Path $LogDir "scheduler-daily-$Timestamp.log"

$VenvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvActivate)) {
    Write-Error "Virtual environment not found at $VenvActivate. Create it first: python -m venv venv"
    exit 1
}

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting daily job. Logging to $LogFile" | Tee-Object -FilePath $LogFile -Append

& $VenvActivate
python -m src.scheduler.jobs daily *>&1 | Tee-Object -FilePath $LogFile -Append
$ExitCode = $LASTEXITCODE

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Daily job finished with exit code $ExitCode." | Tee-Object -FilePath $LogFile -Append

exit $ExitCode
