SET(VISUALIZATION_SOURCES
  OpenGLInterface/HandleGLError.cpp
  OpenGLInterface/OpenGLBufferObjectNames.cpp
  )

ADD_LIBRARY(Visualization
  ${VISUALIZATION_SOURCES}
  )

TARGET_LINK_LIBRARIES(Visualization
  UTILITIES
  )