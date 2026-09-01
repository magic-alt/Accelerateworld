add_executable(aw_cuda_graph cuda_graph.cu)
accelerateworld_configure_target(aw_cuda_graph)

if(ACCELERATEWORLD_ENABLE_TESTING)
  add_test(
    NAME cuda.graph
    COMMAND aw_cuda_graph --elements 65536 --iterations 50
  )
  set_tests_properties(cuda.graph PROPERTIES LABELS "gpu;graph;launch-overhead;benchmark" TIMEOUT 120)
endif()
