SET(VISUALIZATION_SOURCES
  GLFWInterface/GLFWWindow.cpp
  GLUTInterface/GLUTWindow.cpp
  OpenGLInterface/BufferObjectNames.cpp
  OpenGLInterface/BufferObjectParameters.cpp
  OpenGLInterface/CreateOpenGLBuffer.cpp
  OpenGLInterface/DrawPixels.cpp
  OpenGLInterface/HandleGLError.cpp
  )

SET(VISUALIZATION_SOURCES_CU
  CUDAGraphicsResource.cu
  GLFWInterface/MakeAnimatedBitMap.cu
  GLFWInterface/MakeBitMap.cu
  float_to_color.cu
  )

ADD_LIBRARY(Visualization
  ${VISUALIZATION_SOURCES}
  ${VISUALIZATION_SOURCES_CU}
  )

# https://cmake.org/cmake/help/latest/command/target_compile_options.html
# target_compile_options(<target> [BEFORE]<INTERFACE|PBULIC PRIVATE>
# [items1...])
# Adds options to COMPILE_OPTIONS OR INTERFACE_COMPILE_OPTIONS target
# properties.
# Apply --expt-extended-lambda flag only to CUDA source files
# Not needed until we need lambda functions.
#TARGET_COMPILE_OPTIONS(Visualization PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#  --expt-extended-lambda
#  >)

TARGET_LINK_LIBRARIES(Visualization
  Utilities_CU
  Utilities
  ${GLUT_LIBRARIES}
  ${OPENGL_LIBRARIES}
  glfw
  )