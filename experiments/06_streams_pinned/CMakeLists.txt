add_executable(aw_streams_pinned streams_pinned.cu)
accelerateworld_configure_target(aw_streams_pinned)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.streams_pinned
    COMMAND aw_streams_pinned --elements 1048576 --streams 4 --iterations 2
  )
  set_tests_properties(cuda.streams_pinned PROPERTIES LABELS "gpu;streams;transfer;benchmark" TIMEOUT 120)
endif()
