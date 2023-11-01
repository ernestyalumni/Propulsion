# GLFWInterface

## GraphicsInterOp

ADD_EXECUTABLE(GraphicsInterOp
  GLFWInterface/GraphicsInterOp/CalculatePixelValues.cu
  GLFWInterface/GraphicsInterOp/main.cu
  )

TARGET_LINK_LIBRARIES(GraphicsInterOp
  Visualization
  )

SET_TARGET_PROPERTIES(GraphicsInterOp
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/IntegrationTests/Visualization/GLFWInterface")

## Ripple

ADD_EXECUTABLE(Ripple
  GLFWInterface/Ripple/Ripple.cu
  GLFWInterface/Ripple/calculate_ripple_value.cu
  GLFWInterface/Ripple/main.cu
  )

TARGET_LINK_LIBRARIES(Ripple
  Visualization
  )

SET_TARGET_PROPERTIES(Ripple
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/IntegrationTests/Visualization/GLFWInterface")

## SampleCode

ADD_EXECUTABLE(SampleRenderCode
  GLFWInterface/SampleCode/SampleCode.cu
  GLFWInterface/SampleCode/fill_RGB.cu
  GLFWInterface/SampleCode/main.cu
  )

TARGET_LINK_LIBRARIES(SampleRenderCode
  Visualization
  )

SET_TARGET_PROPERTIES(SampleRenderCode
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/IntegrationTests/Visualization/GLFWInterface")

ADD_EXECUTABLE(SampleCode
  GLFWInterface/SampleCode/SampleCode.cu
  GLFWInterface/SampleCode/fill_RGB.cu
  GLFWInterface/SampleCode/main_for_test_sample_code.cu
  )

TARGET_LINK_LIBRARIES(SampleCode
  Visualization
  )

SET_TARGET_PROPERTIES(SampleCode
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/IntegrationTests/Visualization/GLFWInterface")

## SimpleHeat

ADD_EXECUTABLE(SimpleHeat
  GLFWInterface/SimpleHeat/Benchmarking.cu
  GLFWInterface/SimpleHeat/SimpleHeat.cu
  GLFWInterface/SimpleHeat/calculate_temperature.cu
  GLFWInterface/SimpleHeat/main.cu
  )

TARGET_LINK_LIBRARIES(SimpleHeat
  Utilities
  Visualization
  )

# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_SEPARABLE_COMPILATION.html
# CUDA_SEPARABLE_COMPILATION enables separate compilation for all CUDA files
# for the given target.
# It should allow us to compile multiple CUDA source files into separate device
# object files and then link them together into a single executable or library.
# It should be necessary when using device code linking (e.g. __device__ or
# __global__ functions in different translation units).
# This is necessary in this particular case for using extern __constant__.
SET_TARGET_PROPERTIES(SimpleHeat
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET_TARGET_PROPERTIES(SimpleHeat
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/IntegrationTests/Visualization/GLFWInterface")

# GLUTInterface

## Julia Set

ADD_EXECUTABLE(JuliaSet
  #GLUTInterface/JuliaSet/DeviceComplexNumber.cu
  GLUTInterface/JuliaSet/IsInJuliaSet.cu
  GLUTInterface/JuliaSet/JuliaSet.cu
  GLUTInterface/JuliaSet/main.cu
  )

TARGET_LINK_LIBRARIES(JuliaSet
  Visualization
  )

SET_TARGET_PROPERTIES(JuliaSet
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/IntegrationTests/Visualization/GLUTInterface")

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
  RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/IntegrationTests/Visualization")

