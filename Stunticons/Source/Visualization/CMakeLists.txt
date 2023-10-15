SET(VISUALIZATION_SOURCES
  GLUTInterface/GLUTWindow.cpp
  OpenGLInterface/CreateOpenGLBuffer.cpp
  OpenGLInterface/HandleGLError.cpp
  OpenGLInterface/OpenGLBufferObjectNames.cpp
  OpenGLInterface/OpenGLBufferObjectParameters.cpp
  )

SET(VISUALIZATION_SOURCES_CU
  CUDAGraphicsResource.cu
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
  )