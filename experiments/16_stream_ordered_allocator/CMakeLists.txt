add_executable(aw_stream_ordered_allocator stream_ordered_allocator.cu)
accelerateworld_configure_target(aw_stream_ordered_allocator)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.stream_ordered_allocator
    COMMAND aw_stream_ordered_allocator --bytes 1048576 --cycles 8 --rounds 2 --spin-cycles 1000 --release-threshold-mb 64 --pool-max-mb 32 --skip-pressure-probe
  )
  set_tests_properties(cuda.stream_ordered_allocator PROPERTIES LABELS "gpu;correctness;runtime;allocator" TIMEOUT 120)
endif()
