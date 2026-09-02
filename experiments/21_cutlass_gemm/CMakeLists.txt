include(Cutlass)

function(accelerateworld_cutlass_arches_at_least output_name minimum_sm)
  set(result)
  foreach(arch IN LISTS ACCELERATEWORLD_CUDA_ARCHITECTURES)
    if(arch MATCHES "^([0-9]+)")
      set(numeric_arch "${CMAKE_MATCH_1}")
      if(numeric_arch GREATER_EQUAL minimum_sm)
        list(APPEND result "${arch}")
      endif()
    endif()
  endforeach()
  set(${output_name} "${result}" PARENT_SCOPE)
endfunction()

accelerateworld_cutlass_arches_at_least(CUTLASS_SM75_ARCHES 75)
accelerateworld_cutlass_arches_at_least(CUTLASS_SM80_ARCHES 80)
accelerateworld_cutlass_arches_at_least(CUTLASS_SM120_ARCHES 120)

if(CUTLASS_SM75_ARCHES)
  add_executable(aw_cutlass_gemm_simt cutlass_gemm_simt.cu)
  target_link_libraries(aw_cutlass_gemm_simt PRIVATE accelerateworld_cutlass)
  accelerateworld_configure_target(aw_cutlass_gemm_simt)
  set_property(TARGET aw_cutlass_gemm_simt PROPERTY CUDA_ARCHITECTURES "${CUTLASS_SM75_ARCHES}")

  add_executable(aw_cutlass_gemm_sm75 cutlass_gemm_sm75.cu)
  target_link_libraries(aw_cutlass_gemm_sm75 PRIVATE accelerateworld_cutlass)
  accelerateworld_configure_target(aw_cutlass_gemm_sm75)
  set_property(TARGET aw_cutlass_gemm_sm75 PROPERTY CUDA_ARCHITECTURES "${CUTLASS_SM75_ARCHES}")

  if(ACCELERATEWORLD_ENABLE_TESTING)
    add_test(
      NAME cuda.cutlass_gemm_simt
      COMMAND aw_cutlass_gemm_simt --m 128 --n 128 --k 128 --warmup 1 --iterations 1
    )
    set_tests_properties(
      cuda.cutlass_gemm_simt
      PROPERTIES LABELS "gpu;correctness;benchmark;cutlass;simt" TIMEOUT 180
    )

    add_test(
      NAME cuda.cutlass_gemm_sm75
      COMMAND aw_cutlass_gemm_sm75 --m 256 --n 256 --k 256 --warmup 1 --iterations 2
    )
    set_tests_properties(
      cuda.cutlass_gemm_sm75
      PROPERTIES LABELS "gpu;correctness;benchmark;cutlass;tensor-core" TIMEOUT 180
    )
  endif()
endif()

if(CUTLASS_SM80_ARCHES)
  add_executable(aw_cutlass_gemm_sm80 cutlass_gemm_sm80.cu)
  target_link_libraries(aw_cutlass_gemm_sm80 PRIVATE accelerateworld_cutlass)
  accelerateworld_configure_target(aw_cutlass_gemm_sm80)
  set_property(TARGET aw_cutlass_gemm_sm80 PROPERTY CUDA_ARCHITECTURES "${CUTLASS_SM80_ARCHES}")

  if(ACCELERATEWORLD_ENABLE_TESTING)
    add_test(
      NAME cuda.cutlass_gemm_sm80
      COMMAND aw_cutlass_gemm_sm80 --m 256 --n 256 --k 256 --warmup 1 --iterations 2 --mode fp16
    )
    set_tests_properties(
      cuda.cutlass_gemm_sm80
      PROPERTIES LABELS "gpu;correctness;benchmark;cutlass;tensor-core;universal-gemm" TIMEOUT 180
    )
  endif()
endif()

if(CUTLASS_SM120_ARCHES)
  add_executable(aw_cutlass_gemm_sm120 cutlass_gemm_sm120.cu)
  target_link_libraries(aw_cutlass_gemm_sm120 PRIVATE accelerateworld_cutlass)
  accelerateworld_configure_target(aw_cutlass_gemm_sm120)
  set_property(TARGET aw_cutlass_gemm_sm120 PROPERTY CUDA_ARCHITECTURES "${CUTLASS_SM120_ARCHES}")

  if(ACCELERATEWORLD_ENABLE_TESTING)
    add_test(
      NAME cuda.cutlass_gemm_sm120
      COMMAND aw_cutlass_gemm_sm120 --m 256 --n 256 --k 256 --warmup 1 --iterations 2
    )
    set_tests_properties(
      cuda.cutlass_gemm_sm120
      PROPERTIES LABELS "gpu;correctness;benchmark;cutlass;blackwell;universal-gemm" TIMEOUT 180
    )
  endif()
endif()
