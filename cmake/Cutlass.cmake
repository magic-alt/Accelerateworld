include(FetchContent)

set(ACCELERATEWORLD_CUTLASS_VERSION "4.7.0" CACHE STRING "Pinned NVIDIA CUTLASS version")
set(
  ACCELERATEWORLD_CUTLASS_URL
  "https://github.com/NVIDIA/cutlass/archive/refs/tags/v${ACCELERATEWORLD_CUTLASS_VERSION}.tar.gz"
  CACHE STRING
  "Pinned CUTLASS source archive"
)

FetchContent_Declare(
  accelerateworld_cutlass_source
  URL "${ACCELERATEWORLD_CUTLASS_URL}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_GetProperties(accelerateworld_cutlass_source)
if(NOT accelerateworld_cutlass_source_POPULATED)
  FetchContent_Populate(accelerateworld_cutlass_source)
endif()

if(NOT TARGET accelerateworld_cutlass)
  add_library(accelerateworld_cutlass INTERFACE)
  target_include_directories(
    accelerateworld_cutlass
    INTERFACE
      "${accelerateworld_cutlass_source_SOURCE_DIR}/include"
      "${accelerateworld_cutlass_source_SOURCE_DIR}/tools/util/include"
  )
endif()

set(
  ACCELERATEWORLD_CUTLASS_SOURCE_DIR
  "${accelerateworld_cutlass_source_SOURCE_DIR}"
  CACHE INTERNAL
  "CUTLASS source directory populated by FetchContent"
)
