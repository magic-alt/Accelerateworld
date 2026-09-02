add_executable(aw_histogram histogram.cu)
accelerateworld_configure_target(aw_histogram)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.histogram
    COMMAND aw_histogram --elements 262144 --bins 64 --iterations 2 --distribution all
  )
  set_tests_properties(cuda.histogram PROPERTIES LABELS "gpu;correctness;benchmark" TIMEOUT 120)
endif()
