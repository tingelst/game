add_executable(multivector_auto_diff multivector_auto_diff.cpp)
target_link_libraries(multivector_auto_diff
  ${CERES_LIBRARIES})
