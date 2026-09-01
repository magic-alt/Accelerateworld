add_executable(aw_reduction reduction.cu)
accelerateworld_configure_target(aw_reduction)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.reduction
    COMMAND aw_reduction --elements 1048576 --iterations 3
  )
  set_tests_properties(cuda.reduction PROPERTIES LABELS "gpu;correctness;benchmark" TIMEOUT 120)
endif()
