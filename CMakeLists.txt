cmake_minimum_required(VERSION 2.8.3)
project(game)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(hep-ga REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
  include
  ${HEPGA_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  )

add_executable(sphere_fit src/sphere_fit.cpp)
target_link_libraries(sphere_fit ${CERES_LIBRARIES})


