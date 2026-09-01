add_executable(aw_vector_add vector_add.cu)
accelerateworld_configure_target(aw_vector_add)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.vector_add
    COMMAND aw_vector_add --elements 1048576 --iterations 3
  )
  set_tests_properties(cuda.vector_add PROPERTIES LABELS "gpu;correctness" TIMEOUT 60)
endif()
