SET(UTILITIES_CU_SOURCES
  DeviceManagement/GetCUDADeviceProperties.cu
  HandleUnsuccessfulCuBLASCall.cu
  HandleUnsuccessfulCuSparseCall.cu
  HandleUnsuccessfulCudaCall.cu
  )

ADD_LIBRARY(UTILITIES_CU
  ${UTILITIES_CU_SOURCES}
  )

ADD_EXECUTABLE(GetCUDADeviceProperties
  DeviceManagement/GetCUDADeviceProperties_main.cu
  )

TARGET_LINK_LIBRARIES(GetCUDADeviceProperties
  UTILITIES_CU
  )

SET_TARGET_PROPERTIES(GetCUDADeviceProperties
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Utilities/DeviceManagement")
