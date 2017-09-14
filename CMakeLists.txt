cmake_minimum_required(VERSION 2.8.3)
project(game)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(BUILD_TESTING "Enable tests" OFF)
option(BUILD_EXAMPLES "Build examples" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -ftemplate-depth-1200 -Wno-switch -Wno-deprecated-declarations -Wno-int-to-pointer-cast")

# find_package(PythonLibs 2.7.6 REQUIRED)
find_package(PythonLibs 3.5.2 REQUIRED)

find_package(Ceres REQUIRED)
find_package(benchmark REQUIRED)

find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  MESSAGE("-- Building with OpenMP.")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
        include
        third_party/Adept/include
        third_party/pybind11/include/
        third_party/vahlen/include/
        third_party/hep-ga/include/
        ${PYTHON_INCLUDE_DIRS}
        ${EIGEN_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${BENCHMARK_INCLUDE_DIR}
)

if (BUILD_EXAMPLES)
  message("-- Build the examples.")
  add_subdirectory(examples)
else (BUILD_EXAMPLES)
  message("-- Do not build any example.")
endif (BUILD_EXAMPLES)



add_library(adept SHARED 
  third_party/Adept/adept/adept.cpp 
  third_party/Adept/adept/adept_openmp.cpp
  )

# add_executable(benchmark 
#   src/benchmark.cpp
#   )
# target_link_libraries(benchmark
#   adept
#   pthread
#   ${CERES_LIBRARIES}
#   ${BENCHMARK_LIBRARY}
#   )



add_library(cylinder_estimation SHARED 
  src/cylinder_estimation.cpp
  )
target_link_libraries(cylinder_estimation
  ${PYTHON_LIBRARIES}
  ${CERES_LIBRARIES})
set_target_properties(cylinder_estimation PROPERTIES PREFIX "")
set_target_properties(cylinder_estimation PROPERTIES SUFFIX ".so")

add_library(motor_estimation SHARED 
  src/motor_estimation.cpp
  )

target_link_libraries(motor_estimation
  adept
  ${PYTHON_LIBRARIES}
  ${CERES_LIBRARIES}
  )
set_target_properties(motor_estimation PROPERTIES PREFIX "")
set_target_properties(motor_estimation PROPERTIES SUFFIX ".so")


add_library(motor_jacobian SHARED
  src/motor_local_parameterization_jacobian_py.cpp
  )
target_link_libraries(motor_jacobian
  ${PYTHON_LIBRARIES}
  ${CERES_LIBRARIES}
  )
set_target_properties(motor_jacobian PROPERTIES PREFIX "")
set_target_properties(motor_jacobian PROPERTIES SUFFIX ".so")

add_library(motor_estimation_valkenburg_dorst SHARED
  src/motor_estimation_valkenburg_dorst.cpp
  src/vsr/cga_op.cpp
  src/vsr/cga_round.cpp
  )
target_link_libraries(motor_estimation_valkenburg_dorst
  ${PYTHON_LIBRARIES}
  ${CERES_LIBRARIES}
  )
set_target_properties(motor_estimation_valkenburg_dorst PROPERTIES PREFIX "")
set_target_properties(motor_estimation_valkenburg_dorst PROPERTIES SUFFIX ".so")

add_library(autodiff_multivector SHARED
  src/autodiff_multivector.cpp)
target_link_libraries(autodiff_multivector
  adept
  ${PYTHON_LIBRARIES}
  ${CERES_LIBRARIES}
  )
set_target_properties(autodiff_multivector PROPERTIES PREFIX "")
set_target_properties(autodiff_multivector PROPERTIES SUFFIX ".so")

add_library(hyperdual SHARED
  src/test_hyperdual_hessian.cpp)
target_link_libraries(hyperdual
  ${PYTHON_LIBRARIES}
  )
set_target_properties(hyperdual PROPERTIES PREFIX "")
set_target_properties(hyperdual PROPERTIES SUFFIX ".so")

add_library(hyperdual_lines SHARED
  src/hyperdual_lines.cpp)
target_link_libraries(hyperdual_lines
  ${PYTHON_LIBRARIES}
  )
set_target_properties(hyperdual_lines PROPERTIES PREFIX "")
set_target_properties(hyperdual_lines PROPERTIES SUFFIX ".so")

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
  src/vsr/versor_direction_bivector_pybind11.cpp
  src/vsr/versor_dual_plane_pybind11.cpp
  src/vsr/versor_plane_pybind11.cpp
  src/vsr/versor_general_rotor_pybind11.cpp
  src/vsr/versor_tangent_vector_pybind11.cpp
  src/vsr/versor_circle_pybind11.cpp
  src/vsr/versor_sphere_pybind11.cpp
  src/vsr/versor_point_pair_pybind11.cpp
  src/vsr/versor_trivector_pybind11.cpp
  src/vsr/versor_diff_pybind11.cpp
  src/vsr/versor_reciprocal_motor_pybind11.cpp
  src/vsr/versor_euclidean_geometric_algebra_pybind11.cpp
  src/vsr/versor_conformal_geometric_algebra_pybind11.cpp
  src/vsr/versor_e4_pybind11.cpp
  src/vsr/cga_op.cpp
  src/vsr/cga_round.cpp
  )
target_link_libraries(versor_pybind11 ${PYTHON_LIBRARIES})
set_target_properties(versor_pybind11 PROPERTIES PREFIX "")
set_target_properties(versor_pybind11 PROPERTIES SUFFIX ".so")

add_library(linear_solver SHARED src/linear_solver.cpp)
target_link_libraries(linear_solver ${PYTHON_LIBRARIES})
set_target_properties(linear_solver PROPERTIES PREFIX "")
set_target_properties(linear_solver PROPERTIES SUFFIX ".so")

if (BUILD_TESTING)
  message("-- Build the tests.")
  add_subdirectory(test)
else (BUILD_TESTING)
  message("-- Do not build any test.")
endif (BUILD_TESTING)
