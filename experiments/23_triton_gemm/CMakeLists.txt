find_package(CUDAToolkit REQUIRED)

add_executable(aw_triton_cublas_fp16_gemm cublas_fp16_gemm.cu)
accelerateworld_configure_target(aw_triton_cublas_fp16_gemm)
target_link_libraries(aw_triton_cublas_fp16_gemm PRIVATE CUDA::cublas)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.triton_cublas_fp16_gemm
    COMMAND aw_triton_cublas_fp16_gemm --m 128 --n 128 --k 128 --warmup 1 --iterations 2
  )
  set_tests_properties(
    cuda.triton_cublas_fp16_gemm
    PROPERTIES LABELS "gpu;cublas;gemm;triton-reference;tensor-core;benchmark" TIMEOUT 120
  )
endif()
