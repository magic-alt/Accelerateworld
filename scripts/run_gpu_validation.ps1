$ErrorActionPreference = "Stop"

$GpuIndex = if ($env:ACCELERATEWORLD_GPU_INDEX) { $env:ACCELERATEWORLD_GPU_INDEX } else { "0" }
$BuildDir = if ($env:ACCELERATEWORLD_BUILD_DIR) { $env:ACCELERATEWORLD_BUILD_DIR } else { "build/gpu-native" }
$ResultRoot = if ($env:ACCELERATEWORLD_RESULT_ROOT) { $env:ACCELERATEWORLD_RESULT_ROOT } else { "results/gpu-baselines" }

New-Item -ItemType Directory -Force -Path "results" | Out-Null

$argsList = @(
    "scripts/run_gpu_baseline.py",
    "--gpu-index", $GpuIndex,
    "--build-dir", $BuildDir,
    "--result-root", $ResultRoot
)

if ($env:ACCELERATEWORLD_SKIP_SANITIZER -eq "1") {
    $argsList += "--skip-sanitizer"
}

if ($env:ACCELERATEWORLD_ALLOW_NON_RTX -eq "1") {
    $argsList += "--allow-non-rtx"
}

python @argsList 2>&1 | Tee-Object -FilePath "results/gpu-validation.log"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
