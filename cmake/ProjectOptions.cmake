function(accelerateworld_configure_target target_name)
  target_include_directories(
    ${target_name}
    PRIVATE
      "${PROJECT_SOURCE_DIR}/include"
  )

  target_compile_features(${target_name} PRIVATE cxx_std_17 cuda_std_17)

  if(MSVC)
    target_compile_options(
      ${target_name}
      PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
  else()
    target_compile_options(
      ${target_name}
      PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
  endif()

  if(ACCELERATEWORLD_ENABLE_FAST_MATH)
    target_compile_options(
      ${target_name}
      PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
  endif()

  set_target_properties(
    ${target_name}
    PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
  )
endfunction()
