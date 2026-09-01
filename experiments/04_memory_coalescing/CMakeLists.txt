add_executable(aw_memory_coalescing memory_coalescing.cu)
accelerateworld_configure_target(aw_memory_coalescing)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.memory_coalescing
    COMMAND aw_memory_coalescing --elements 262144 --stride 32 --iterations 3
  )
  set_tests_properties(cuda.memory_coalescing PROPERTIES LABELS "gpu;memory;benchmark" TIMEOUT 120)
endif()
