cmake_minimum_required(VERSION 2.8.3)
project(game)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


find_package(Catch)
find_package(Boost REQUIRED COMPONENTS python numpy)
find_package(PythonLibs 2.7 REQUIRED)

find_package(hep-ga REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
  include
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${HEPGA_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${CATCH_INCLUDE_DIR}
  )

add_executable(sphere_fit src/sphere_fit.cpp)
target_link_libraries(sphere_fit ${CERES_LIBRARIES})

add_library(hello_world SHARED src/hello_world.cpp)
target_link_libraries(hello_world ${CERES_LIBRARIES} ${Boost_LIBRARIES})

enable_testing(true)
file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test/*.cpp)

#Run through each source
foreach(TEST_SRC ${TEST_SRCS})
  #Extract the filename without an extension (NAME_WE)
  get_filename_component(TEST ${TEST_SRC} NAME_WE)

  #Add compile target
  add_executable(${TEST} ${TEST_SRC})

  #Finally add it to test execution -
  add_test(NAME ${TEST}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build/${TEST} )
endforeach(TEST_SRC)
