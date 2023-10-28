SET(VISUALIZATION_SOURCES
  GLFWInterface/GLFWWindow.cpp
  GLUTInterface/GLUTWindow.cpp
  GLFWInterface/MakeBitMap.cu
  OpenGLInterface/BufferObjectNames.cpp
  OpenGLInterface/BufferObjectParameters.cpp
  OpenGLInterface/CreateOpenGLBuffer.cpp
  OpenGLInterface/DrawPixels.cpp
  OpenGLInterface/HandleGLError.cpp
  )

SET(VISUALIZATION_SOURCES_CU
  CUDAGraphicsResource.cu
  float_to_color.cu
  )

ADD_LIBRARY(Visualization
  ${VISUALIZATION_SOURCES}
  ${VISUALIZATION_SOURCES_CU}
  )

TARGET_LINK_LIBRARIES(Visualization
  Utilities_CU
  UTILITIES
  ${GLUT_LIBRARIES}
  ${OPENGL_LIBRARIES}
  glfw
  )