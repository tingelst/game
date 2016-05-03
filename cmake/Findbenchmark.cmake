# Findbenchmark.cmake
# - Try to find benchmark
#
# The following variables are optionally searched for defaults
#  BENCHMARK_ROOT_DIR:  Base directory where all benchmark components are found
#
# Once done this will define
#  BENCHMARK_FOUND - System has benchmark
#  BENCHMARK_INCLUDE_DIRS - The benchmark include directories
#  BENCHMARK_LIBRARIES - The libraries needed to use benchmark

set(BENCHMARK_ROOT_DIR "" CACHE PATH "Folder containing benchmark")

find_path(BENCHMARK_INCLUDE_DIR "benchmark/benchmark.h"
  PATHS ${BENCHMARK_ROOT_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)
find_path(BENCHMARK_INCLUDE_DIR "benchmark/benchmark.h")

find_library(BENCHMARK_LIBRARY NAMES "benchmark"
  PATHS ${BENCHMARK_ROOT_DIR}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH)
find_library(BENCHMARK_LIBRARY NAMES "benchmark")

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set BENCHMARK_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(benchmark FOUND_VAR BENCHMARK_FOUND
  REQUIRED_VARS BENCHMARK_LIBRARY
  BENCHMARK_INCLUDE_DIR)

if(BENCHMARK_FOUND)
  set(BENCHMARK_LIBRARIES ${BENCHMARK_LIBRARY})
  set(BENCHMARK_INCLUDE_DIRS ${BENCHMARK_INCLUDE_DIR})
endif()

mark_as_advanced(BENCHMARK_INCLUDE_DIR BENCHMARK_LIBRARY)
