# Get the base directory (where this script is located)
$BASE_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$MOD_DIR = Join-Path $BASE_DIR "modules"

# Change to modules directory
cd $MOD_DIR

# Start all components in background
$processes = @()

# Function to start a process
function Start-PythonModule {
    param (
        [string]$ModuleName
    )
    
    Write-Host "Starting $ModuleName..."
    $process = Start-Process -FilePath "python" -ArgumentList "$ModuleName.py" -PassThru -WindowStyle Normal
    return $process
}

# Start all required modules
$modules = @("input", "text_vap", "asr", "dialogue", "tts", "output")
foreach ($module in $modules) {
    $processes += Start-PythonModule -ModuleName $module
}

Write-Host "All processes started. Press Ctrl+C to stop all processes."

try {
    # Keep script running until Ctrl+C
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    # Stop all processes when script exits
    foreach ($process in $processes) {
        if (!$process.HasExited) {
            Write-Host "Stopping process $($process.Id)..."
            Stop-Process -Id $process.Id -Force
        }
    }
    Write-Host "All processes terminated"
}