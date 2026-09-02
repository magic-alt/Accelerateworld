find_package(CUDAToolkit REQUIRED)

add_executable(aw_fp4_gemm fp4_gemm.cu)
accelerateworld_configure_target(aw_fp4_gemm)
target_link_libraries(aw_fp4_gemm PRIVATE CUDA::cublasLt CUDA::cublas)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.fp4_gemm
    COMMAND aw_fp4_gemm --m 256 --n 256 --k 256 --heuristics 8 --warmup 1 --iterations 2 --workspace-mb 8 --mode all
  )
  set_tests_properties(cuda.fp4_gemm PROPERTIES LABELS "gpu;cublaslt;gemm;fp4;blackwell;block-scale;tensor-core;benchmark" TIMEOUT 120)
endif()
