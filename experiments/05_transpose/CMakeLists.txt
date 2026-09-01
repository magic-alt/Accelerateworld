add_executable(aw_transpose transpose.cu)
accelerateworld_configure_target(aw_transpose)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.transpose
    COMMAND aw_transpose --size 256 --iterations 3
  )
  set_tests_properties(cuda.transpose PROPERTIES LABELS "gpu;shared-memory;benchmark" TIMEOUT 120)
endif()
