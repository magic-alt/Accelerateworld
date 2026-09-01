$ErrorActionPreference = "Stop"

$Arch = if ($env:ACCELERATEWORLD_CUDA_ARCH) { $env:ACCELERATEWORLD_CUDA_ARCH } else { "120" }
$BuildDir = if ($env:ACCELERATEWORLD_BUILD_DIR) { $env:ACCELERATEWORLD_BUILD_DIR } else { "build/gpu" }
$ResultDir = if ($env:ACCELERATEWORLD_RESULT_DIR) { $env:ACCELERATEWORLD_RESULT_DIR } else { "results" }

New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null
$LogFile = Join-Path $ResultDir "gpu-validation.log"

$transcriptStarted = $false
try {
    Start-Transcript -Path $LogFile -Force | Out-Null
    $transcriptStarted = $true

    Write-Host "== Accelerateworld GPU validation =="
    Write-Host "CUDA architecture: $Arch"

    nvidia-smi
    nvcc --version

    cmake -S . -B $BuildDir -G Ninja `
        -DCMAKE_BUILD_TYPE=Release `
        "-DACCELERATEWORLD_CUDA_ARCHITECTURES=$Arch"
    cmake --build $BuildDir --parallel

    ctest --test-dir $BuildDir --output-on-failure

    compute-sanitizer --tool memcheck --error-exitcode 99 `
        "$BuildDir/bin/aw_vector_add.exe" --elements 1048576 --iterations 2
    compute-sanitizer --tool memcheck --error-exitcode 99 `
        "$BuildDir/bin/aw_matmul.exe" --size 128 --iterations 2
    compute-sanitizer --tool racecheck --error-exitcode 99 `
        "$BuildDir/bin/aw_matmul.exe" --size 128 --iterations 1
    compute-sanitizer --tool initcheck --error-exitcode 99 `
        "$BuildDir/bin/aw_vector_add.exe" --elements 1048576 --iterations 1
    compute-sanitizer --tool synccheck --error-exitcode 99 `
        "$BuildDir/bin/aw_matmul.exe" --size 128 --iterations 1

    & "$BuildDir/bin/aw_vector_add.exe" --elements 33554432 --iterations 50
    & "$BuildDir/bin/aw_matmul.exe" --size 1024 --iterations 10

    Write-Host "GPU validation: PASS"
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}
