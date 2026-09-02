add_executable(aw_prefix_scan prefix_scan.cu)
accelerateworld_configure_target(aw_prefix_scan)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.prefix_scan
    COMMAND aw_prefix_scan --elements 1048576 --iterations 3
  )
  set_tests_properties(cuda.prefix_scan PROPERTIES LABELS "gpu;correctness;benchmark;scan" TIMEOUT 180)
endif()
