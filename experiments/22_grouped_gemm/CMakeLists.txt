include(Cutlass)

function(accelerateworld_grouped_arches_at_least output_name minimum_sm)
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

accelerateworld_grouped_arches_at_least(GROUPED_GEMM_SM75_ARCHES 75)

if(GROUPED_GEMM_SM75_ARCHES)
  add_executable(aw_grouped_gemm grouped_gemm.cu)
  target_link_libraries(aw_grouped_gemm PRIVATE accelerateworld_cutlass)
  accelerateworld_configure_target(aw_grouped_gemm)
  set_property(TARGET aw_grouped_gemm PROPERTY CUDA_ARCHITECTURES "${GROUPED_GEMM_SM75_ARCHES}")

  if(ACCELERATEWORLD_ENABLE_TESTING)
    add_test(
      NAME cuda.grouped_gemm
      COMMAND aw_grouped_gemm --workload heterogeneous --groups 6 --warmup 0 --iterations 1 --mode device
    )
    set_tests_properties(
      cuda.grouped_gemm
      PROPERTIES LABELS "gpu;correctness;benchmark;cutlass;grouped-gemm;persistent;tensor-core" TIMEOUT 180
    )
  endif()
endif()
