$ErrorActionPreference = "Stop"

Write-Host "== Accelerateworld environment check =="

$commands = @("nvidia-smi", "nvcc", "cmake")
foreach ($command in $commands) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "$command is not available on PATH."
    }
}

nvidia-smi
Write-Host ""
nvcc --version
Write-Host ""
cmake --version | Select-Object -First 1

Write-Host ""
Write-Host "Environment check: PASS"
