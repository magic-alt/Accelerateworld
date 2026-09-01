param(
    [string]$BuildDir = "build/gpu",
    [string]$ResultDir = "results"
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null
$EnvironmentFile = Join-Path $ResultDir "rtx5060-environment.txt"
$DeviceQueryFile = Join-Path $ResultDir "rtx5060-device-query.txt"
$NativeJson = Join-Path $ResultDir "rtx5060-native.json"
$SummaryJson = Join-Path $ResultDir "rtx5060-baseline-summary.json"

foreach ($Command in @("nvidia-smi", "nvcc", "cmake", "python", "compute-sanitizer")) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Command"
    }
}

$GitCommit = (git rev-parse HEAD).Trim()
$EnvironmentLines = @(
    "== RTX 5060 baseline environment =="
    "UTC: $([DateTime]::UtcNow.ToString('o'))"
    "git: $GitCommit"
    ""
    (nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader | Out-String).TrimEnd()
    ""
    (nvcc --version | Out-String).TrimEnd()
    ""
    (cmake --version | Out-String).TrimEnd()
    ""
    (python --version 2>&1 | Out-String).TrimEnd()
)
$EnvironmentLines | Set-Content -Encoding UTF8 $EnvironmentFile
$EnvironmentLines | ForEach-Object { Write-Host $_ }

$EnvironmentText = Get-Content -Raw $EnvironmentFile
if ($EnvironmentText -notmatch "RTX 5060") {
    throw "This baseline must run on an NVIDIA GeForce RTX 5060."
}

$env:ACCELERATEWORLD_CUDA_ARCH = "120"
$env:ACCELERATEWORLD_BUILD_DIR = $BuildDir
$env:ACCELERATEWORLD_RESULT_DIR = $ResultDir

& "$PSScriptRoot/run_gpu_validation.ps1"
if ($LASTEXITCODE -ne 0) { throw "Native GPU validation failed with exit code $LASTEXITCODE" }

$DeviceQueryExe = Join-Path $BuildDir "bin/aw_device_query.exe"
$DeviceOutput = & $DeviceQueryExe 2>&1 | Tee-Object -FilePath $DeviceQueryFile
if ($LASTEXITCODE -ne 0) { throw "Device query failed with exit code $LASTEXITCODE" }
$DeviceText = ($DeviceOutput | Out-String)
if ($DeviceText -notmatch "RTX 5060") { throw "Device query did not report RTX 5060." }
if ($DeviceText -notmatch "Compute capability: 12\.0") { throw "Device query did not report Compute Capability 12.0." }

python scripts/run_benchmark_suite.py --build-dir $BuildDir --output $NativeJson
if ($LASTEXITCODE -ne 0) { throw "Native benchmark suite failed with exit code $LASTEXITCODE" }

python scripts/verify_baseline_evidence.py --result-dir $ResultDir
if ($LASTEXITCODE -ne 0) { throw "Baseline evidence verification failed with exit code $LASTEXITCODE" }

$NativeReport = Get-Content -Raw $NativeJson | ConvertFrom-Json
$Summary = [ordered]@{
    schema_version = 1
    status = "PASS"
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    git_commit = $GitCommit
    target = "NVIDIA GeForce RTX 5060"
    cuda_architecture = "sm_120"
    native_success = [bool]$NativeReport.success
    ai_validation_included = $false
    evidence = @(
        "rtx5060-environment.txt",
        "rtx5060-device-query.txt",
        "gpu-validation.log",
        "rtx5060-native.json"
    )
}
$Summary | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $SummaryJson
$Summary | ConvertTo-Json -Depth 5 | Write-Host
Write-Host "RTX 5060 native baseline: PASS"
Write-Host "For Triton/AI runtime validation, run scripts/run_rtx5060_baseline.sh --with-ai from Linux/WSL2."
