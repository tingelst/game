cmake_minimum_required(VERSION 2.8.3)
project(game)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -ftemplate-depth-1200 -Wno-switch -Wno-deprecated-declarations -Wno-int-to-pointer-cast")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -stdlib=libc++ -ftemplate-depth-1200 -Wno-switch -Wno-deprecated-declarations -Wno-int-to-pointer-cast")


# find_package(hep-ga REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)
find_package(PythonLibs 2.7.10 REQUIRED)

set(Ceres_DIR /Users/lars/devel/game_ws/game/third_party/ceres-solver/cmake)
find_package(Ceres REQUIRED)

find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  MESSAGE("-- Building with OpenMP.")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
        include
        ../hep-ga/include/
        third_party/Adept/include
        third_party/pybind11/include/
        ${Boost_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
)

# add_library(sphere_fit SHARED src/sphere_fit.cpp)
# target_link_libraries(sphere_fit ${CERES_LIBRARIES} ${Boost_LIBRARIES})

#add_executable(rotor_estimation src/rotor_estimation.cpp)
#target_link_libraries(rotor_estimation ${CERES_LIBRARIES})
#
# add_executable(rotor_bivector_generator_estimation src/rotor_bivector_generator_estimation.cpp)
# target_link_libraries(rotor_bivector_generator_estimation ${CERES_LIBRARIES})

# add_library(rotor_estimation SHARED src/rotor_estimation_wrapper.cpp)
# target_link_libraries(rotor_estimation ${CERES_LIBRARIES} ${Boost_LIBRARIES})

add_library(adept SHARED 
  third_party/Adept/adept/adept.cpp 
  third_party/Adept/adept/adept_openmp.cpp
  )

# add_library(lds_calibration SHARED src/lds_calibration.cpp)
# target_link_libraries(lds_calibration
#   adept
#   ${PYTHON_LIBRARIES}
#   ${CERES_LIBRARIES}
#   ${Boost_LIBRARIES})

add_library(motor_estimation SHARED 
  src/motor_estimation.cpp
  third_party/Adept/adept/adept.cpp 
  third_party/Adept/adept/adept_openmp.cpp
  )

target_link_libraries(motor_estimation 
  ${PYTHON_LIBRARIES}
  ${CERES_LIBRARIES})
set_target_properties(motor_estimation PROPERTIES PREFIX "")
set_target_properties(motor_estimation PROPERTIES SUFFIX ".so")

add_library(hand_eye_calibration SHARED
  src/hand_eye_calibration.cpp)

target_link_libraries(hand_eye_calibration
  ${PYTHON_LIBRARIES}
  ${CERES_LIBRARIES})
set_target_properties(hand_eye_calibration PROPERTIES PREFIX "")
set_target_properties(hand_eye_calibration PROPERTIES SUFFIX ".so")



add_executable(adept_rotor_diff_example
  src/adept_rotor_diff_example.cpp
  third_party/Adept/adept/adept.cpp 
  )


add_library(rigid_body_motion_estimation SHARED src/rigid_body_motion_estimation_wrapper.cpp)
target_link_libraries(rigid_body_motion_estimation ${CERES_LIBRARIES} ${Boost_LIBRARIES})

# add_library(kinematic_calibration SHARED
#         src/versor_kinematic_calibration_wrapper.cpp)
# set_target_properties(kinematic_calibration PROPERTIES COMPILE_FLAGS "-save-temps ")
# target_link_libraries(kinematic_calibration
#         ${CERES_LIBRARIES}
#         ${Boost_LIBRARIES})

add_library(versor_cga_op SHARED src/vsr/cga_op.cpp src/vsr/cga_round.cpp src/vsr/python_interface_cga_op.cpp)
target_link_libraries(versor_cga_op ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})

# add_library(versor SHARED src/vsr/python_interface.cpp)
# target_link_libraries(versor ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})

add_library(versorninja SHARED ./src/test_bindings.cpp)
#add_library(versorninja SHARED ./src/vsr/init1.cpp src/vsr/vsr_pybind11.cpp)
target_link_libraries(versorninja ${PYTHON_LIBRARIES})
set_target_properties(versorninja PROPERTIES PREFIX "")
set_target_properties(versorninja PROPERTIES SUFFIX ".so")

add_library(versor_pybind11 SHARED
  src/vsr/versor_pybind11.cpp
  src/vsr/versor_vector_pybind11.cpp
  src/vsr/versor_bivector_pybind11.cpp
  src/vsr/versor_rotor_pybind11.cpp
  src/vsr/versor_point_pybind11.cpp
  src/vsr/versor_dual_line_pybind11.cpp
  src/vsr/versor_line_pybind11.cpp
  src/vsr/versor_translator_pybind11.cpp
  src/vsr/versor_motor_pybind11.cpp
  src/vsr/versor_origin_pybind11.cpp
  src/vsr/versor_infinity_pybind11.cpp
  src/vsr/versor_direction_vector_pybind11.cpp
  src/vsr/versor_dual_plane_pybind11.cpp
  src/vsr/versor_plane_pybind11.cpp
  src/vsr/versor_general_rotor_pybind11.cpp
  src/vsr/versor_tangent_vector_pybind11.cpp
  src/vsr/versor_circle_pybind11.cpp
  src/vsr/versor_sphere_pybind11.cpp
  src/vsr/versor_point_pair_pybind11.cpp
  src/vsr/cga_op.cpp
  src/vsr/cga_round.cpp
  )
target_link_libraries(versor_pybind11 ${PYTHON_LIBRARIES})
set_target_properties(versor_pybind11 PROPERTIES PREFIX "")
set_target_properties(versor_pybind11 PROPERTIES SUFFIX ".so")

#add_executable(test_versor src/versor_rotor_estimation.cpp)
#target_link_libraries(test_versor ${CERES_LIBRARIES})

# add_library(hello_world SHARED src/hello_world.cpp)
# target_link_libraries(hello_world ${CERES_LIBRARIES} ${Boost_LIBRARIES})


# enable_testing(false)
# file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test/*.cpp)

# add_executable(test_speed_no_ceres test/test_speed.cpp)

#Run through each source
# foreach(TEST_SRC ${TEST_SRCS})
#   #Extract the filename without an extension (NAME_WE)
#   get_filename_component(TEST ${TEST_SRC} NAME_WE)

#   #Add compile target
#   add_executable(${TEST} ${TEST_SRC} ./third_party/Adept/adept/adept.cpp)
#       target_link_libraries(${TEST} ${CERES_LIBRARIES})

#   #Finally add it to test execution -
#   add_test(NAME ${TEST}
#     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build/${TEST} )
# endforeach(TEST_SRC)
