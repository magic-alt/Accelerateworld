add_executable(aw_device_query device_query.cu)
accelerateworld_configure_target(aw_device_query)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(NAME cuda.device_query COMMAND aw_device_query)
  set_tests_properties(cuda.device_query PROPERTIES LABELS "gpu;smoke" TIMEOUT 30)
endif()
