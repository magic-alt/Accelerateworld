find_package(CUDAToolkit REQUIRED)

add_executable(aw_cublaslt_autotune cublaslt_autotune.cu)
accelerateworld_configure_target(aw_cublaslt_autotune)
target_link_libraries(aw_cublaslt_autotune PRIVATE CUDA::cublas CUDA::cublasLt)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.cublaslt_autotune
    COMMAND aw_cublaslt_autotune --m 128 --n 128 --k 128 --heuristics 4 --warmup 1 --iterations 2 --workspace-mb 8 --no-cache
  )
  set_tests_properties(cuda.cublaslt_autotune PROPERTIES LABELS "gpu;cublaslt;gemm;autotune;benchmark" TIMEOUT 120)
endif()
