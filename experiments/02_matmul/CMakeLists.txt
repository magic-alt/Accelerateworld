add_executable(aw_matmul matmul.cu)
accelerateworld_configure_target(aw_matmul)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.matmul
    COMMAND aw_matmul --size 128 --iterations 3
  )
  set_tests_properties(cuda.matmul PROPERTIES LABELS "gpu;correctness;benchmark" TIMEOUT 120)
endif()
