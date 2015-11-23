cmake_minimum_required(VERSION 2.8.3)
project(game)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -ftemplate-depth-1200 -Wno-switch -Wno-deprecated-declarations -Wno-int-to-pointer-cast")

find_package(Boost REQUIRED COMPONENTS python numpy)
find_package(PythonLibs 2.7 REQUIRED)

find_package(hep-ga REQUIRED)

find_package(Ceres REQUIRED)

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


add_library(motor_estimation SHARED src/motor_estimation.cpp)
target_link_libraries(motor_estimation ${CERES_LIBRARIES} ${Boost_LIBRARIES})

add_library(rigid_body_motion_estimation SHARED src/rigid_body_motion_estimation_wrapper.cpp)
target_link_libraries(rigid_body_motion_estimation ${CERES_LIBRARIES} ${Boost_LIBRARIES})

add_library(kinematic_calibration SHARED
        src/versor_kinematic_calibration_wrapper.cpp)
set_target_properties(kinematic_calibration PROPERTIES COMPILE_FLAGS "-save-temps ")
target_link_libraries(kinematic_calibration
        ${CERES_LIBRARIES}
        ${Boost_LIBRARIES})

add_library(pso SHARED src/pso_rbm_estimation.cpp)
target_link_libraries(pso ${CERES_LIBRARIES} ${Boost_LIBRARIES})

add_library(versor SHARED src/vsr/python_interface.cpp)
target_link_libraries(versor ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})

#add_executable(test_versor src/versor_rotor_estimation.cpp)
#target_link_libraries(test_versor ${CERES_LIBRARIES})

add_library(hello_world SHARED src/hello_world.cpp)
target_link_libraries(hello_world ${CERES_LIBRARIES} ${Boost_LIBRARIES})


enable_testing(true)
file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test/*.cpp)

add_executable(test_speed_no_ceres test/test_speed.cpp)

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
