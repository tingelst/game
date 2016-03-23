enable_testing()
file(GLOB TEST_SRCS *.cpp)

# Run through each source
foreach(TEST_SRC ${TEST_SRCS})
  #Extract the filename without an extension (NAME_WE)
  get_filename_component(TEST ${TEST_SRC} NAME_WE)

  #Add compile target
  add_executable(${TEST} ${TEST_SRC})
  target_link_libraries(${TEST} adept ${CERES_LIBRARIES})
  
  #Finally add it to test execution -
  add_test(NAME ${TEST}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_SOURCE_DIR}/build/test/${TEST} )
endforeach(TEST_SRC)
