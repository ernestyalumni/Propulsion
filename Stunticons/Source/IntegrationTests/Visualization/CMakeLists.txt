# GLUTInterface

## Julia Set

ADD_EXECUTABLE(JuliaSet
  GLUTInterface/JuliaSet/DeviceComplexNumber.cu
  GLUTInterface/JuliaSet/JuliaSet.cu
  GLUTInterface/JuliaSet/main.cu
  )

TARGET_LINK_LIBRARIES(JuliaSet
  Visualization
  )

SET_TARGET_PROPERTIES(JuliaSet
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Visualization")

## StripedPattern

ADD_EXECUTABLE(StripedPattern
  GLUTInterface/simpleCUDA2GL/StripedPattern.cu
  # CMake required this file, being that it included StripedPattern.h and that
  # header file needed cuda_runtime.h, to be suffix'ed .cu because otherwise, if
  # suffixed .cpp, it can't find the header. Consider finding ways to include
  # cuda_runtime.h manually.
  GLUTInterface/simpleCUDA2GL/main.cu
  GLUTInterface/simpleCUDA2GL/simpleCUDA2GL.cu
  )

TARGET_LINK_LIBRARIES(StripedPattern
  Visualization
  )

SET_TARGET_PROPERTIES(StripedPattern
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Visualization")

