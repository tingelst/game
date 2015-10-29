cmake_minimum_required(VERSION 2.8.3)
project(game)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


find_package(Catch)
find_package(Boost REQUIRED COMPONENTS python numpy)
find_package(PythonLibs 2.7 REQUIRED)

find_package(hep-ga REQUIRED)

find_package(Ceres REQUIRED)
find_package(Armadillo)
find_package(Mantella)

include_directories(
        include
        third_party/Adept/include
        ${Boost_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${HEPGA_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}
        ${CATCH_INCLUDE_DIR}
        ${ARMADILLO_INCLUDE_DIR}
        ${MANTELLA_INCLUDE_DIR}
)

add_library(sphere_fit SHARED src/sphere_fit.cpp)
target_link_libraries(sphere_fit ${CERES_LIBRARIES} ${Boost_LIBRARIES})

#add_executable(rotor_estimation src/rotor_estimation.cpp)
#target_link_libraries(rotor_estimation ${CERES_LIBRARIES})
#
add_executable(rotor_bivector_generator_estimation src/rotor_bivector_generator_estimation.cpp)
target_link_libraries(rotor_bivector_generator_estimation ${CERES_LIBRARIES})

add_library(rotor_estimation SHARED src/rotor_estimation_wrapper.cpp)
target_link_libraries(rotor_estimation ${CERES_LIBRARIES} ${Boost_LIBRARIES})

add_library(rigid_body_motion_estimation SHARED src/rigid_body_motion_estimation_wrapper.cpp)
target_link_libraries(rigid_body_motion_estimation ${CERES_LIBRARIES} ${Boost_LIBRARIES})


add_library(hello_world SHARED src/hello_world.cpp)
target_link_libraries(hello_world ${CERES_LIBRARIES} ${Boost_LIBRARIES})

add_executable(mantella_test src/mantella_test.cpp)
target_link_libraries(mantella_test ${ARMADILLO_LIBRARIES} ${MANTELLA_LIBRARIES})

enable_testing(true)
file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test/*.cpp)

#Run through each source
foreach(TEST_SRC ${TEST_SRCS})
  #Extract the filename without an extension (NAME_WE)
  get_filename_component(TEST ${TEST_SRC} NAME_WE)

  #Add compile target
  add_executable(${TEST} ${TEST_SRC})
      target_link_libraries(${TEST} ${CERES_LIBRARIES})

  #Finally add it to test execution -
  add_test(NAME ${TEST}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build/${TEST} )
endforeach(TEST_SRC)
