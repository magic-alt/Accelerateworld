add_executable(aw_tensor_core_wmma tensor_core_wmma.cu)
accelerateworld_configure_target(aw_tensor_core_wmma)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.tensor_core_wmma
    COMMAND aw_tensor_core_wmma --size 128 --iterations 3
  )
  set_tests_properties(cuda.tensor_core_wmma PROPERTIES LABELS "gpu;tensor-core;wmma;benchmark" TIMEOUT 120)
endif()
