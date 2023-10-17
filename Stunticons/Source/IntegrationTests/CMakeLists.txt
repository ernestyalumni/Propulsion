ENABLE_TESTING()

SET(TEST_CU_SOURCES
  Visualization/CUDAGraphicsResource_tests.cu
  )

ADD_EXECUTABLE(IntegrationCheck
  ${TEST_CU_SOURCES}
  )

TARGET_LINK_LIBRARIES(IntegrationCheck
  Visualization
  gtest_main
  )

SET_TARGET_PROPERTIES(IntegrationCheck
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

ADD_EXECUTABLE(SimpleGLExample
  Visualization/SimpleGLExample.cu
  )

TARGET_LINK_LIBRARIES(SimpleGLExample
  Utilities_CU
  Visualization
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  )

SET_TARGET_PROPERTIES(SimpleGLExample
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

ADD_SUBDIRECTORY(Visualization)