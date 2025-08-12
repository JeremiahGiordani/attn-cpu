cmake_minimum_required(VERSION 3.16)
project(attn_cpu LANGUAGES CXX)

option(ATTN_BUILD_PYTHON "Build Python extension" ON)
option(ATTN_USE_OPENMP   "Enable OpenMP (not required for M0)" OFF)
option(ATTN_NATIVE_OPT   "Enable -march=native" ON)

# Silence pybind11 message; prefer modern FindPython behavior
set(PYBIND11_FINDPYTHON NEW)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# Compile flags
if (MSVC)
  add_compile_options(/O2 /fp:fast)
else()
  add_compile_options(-O3 -fno-omit-frame-pointer)
  if (ATTN_NATIVE_OPT)
    add_compile_options(-march=native)
  endif()
endif()

# Optional OpenMP
if (ATTN_USE_OPENMP)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: enabling")
    add_compile_definitions(ATTN_USE_OPENMP=1)
  endif()
endif()

# Third-party (vendored)
add_subdirectory(third_party/pybind11)
# add_subdirectory(third_party/llama.cpp)  # not used yet in M0

# ---------- Library target ----------
add_library(attn STATIC
  src/attn_dense.cpp
  src/attn_block.cpp
  src/attn_mha.cpp 
)

target_include_directories(attn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (ATTN_USE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(attn PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------- Python module ----------
if (ATTN_BUILD_PYTHON)
  # You moved the binding to src/bindings.cpp
  pybind11_add_module(attn_cpu src/bindings.cpp)
  target_link_libraries(attn_cpu PRIVATE attn)

  # Put the built .so in the build/ directory root (optional)
  set_target_properties(attn_cpu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()
