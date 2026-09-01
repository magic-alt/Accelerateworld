find_package(CUDAToolkit REQUIRED)

add_executable(aw_cublas_gemm cublas_gemm.cu)
accelerateworld_configure_target(aw_cublas_gemm)
target_link_libraries(aw_cublas_gemm PRIVATE CUDA::cublas)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.cublas_gemm
    COMMAND aw_cublas_gemm --size 128 --iterations 3
  )
  set_tests_properties(cuda.cublas_gemm PROPERTIES LABELS "gpu;cublas;gemm;benchmark" TIMEOUT 120)
endif()
