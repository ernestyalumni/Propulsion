SET(VISUALIZATION_SOURCES
  OpenGLInterface/CreateOpenGLBuffer.cpp
  OpenGLInterface/HandleGLError.cpp
  OpenGLInterface/OpenGLBufferObjectNames.cpp
  OpenGLInterface/OpenGLBufferObjectParameters.cpp
  )

ADD_LIBRARY(Visualization
  ${VISUALIZATION_SOURCES}
  )

TARGET_LINK_LIBRARIES(Visualization
  UTILITIES
  )