add_executable(aw_async_memory_pool async_memory_pool.cu)
accelerateworld_configure_target(aw_async_memory_pool)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.async_memory_pool
    COMMAND aw_async_memory_pool --bytes 1048576 --cycles 8 --rounds 3 --streams 2 --release-threshold-mb 16
  )
  set_tests_properties(cuda.async_memory_pool PROPERTIES LABELS "gpu;correctness;benchmark;runtime" TIMEOUT 120)
endif()
