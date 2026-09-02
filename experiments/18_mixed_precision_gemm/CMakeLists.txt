find_package(CUDAToolkit REQUIRED)

add_executable(aw_mixed_precision_gemm mixed_precision_gemm.cu)
accelerateworld_configure_target(aw_mixed_precision_gemm)
target_link_libraries(aw_mixed_precision_gemm PRIVATE CUDA::cublasLt CUDA::cublas)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.mixed_precision_gemm
    COMMAND aw_mixed_precision_gemm --m 256 --n 256 --k 256 --heuristics 8 --warmup 1 --iterations 2 --workspace-mb 8 --mode all
  )
  set_tests_properties(cuda.mixed_precision_gemm PROPERTIES LABELS "gpu;cublaslt;gemm;mixed-precision;tensor-core;benchmark" TIMEOUT 120)
endif()
