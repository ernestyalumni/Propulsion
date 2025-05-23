SET(UTILITIES_CU_SOURCES
  DeviceManagement/GetCUDADeviceProperties.cu
  DeviceManagement/GetAndSetDevice.cu
  HandleUnsuccessfulCuBLASCall.cu
  HandleUnsuccessfulCuSparseCall.cu
  HandleUnsuccessfulCUDACall.cu
  )

ADD_LIBRARY(Utilities_CU
  ${UTILITIES_CU_SOURCES}
  )

ADD_LIBRARY(Utilities
  FileIO/FilePath.cpp
  FileIO/ReadMatrixMarketFile.cpp
  FileIO/TurbulentFlow/Configuration.cpp
  FileIO/TurbulentFlow/ParseGeometryFile.cpp
  FileIO/TurbulentFlow/ReadConfiguration.cpp
  )

TARGET_LINK_LIBRARIES(Utilities
  ALGEBRA
  )

ADD_EXECUTABLE(GetCUDADeviceProperties
  DeviceManagement/GetCUDADeviceProperties_main.cu
  )

TARGET_LINK_LIBRARIES(GetCUDADeviceProperties
  Utilities_CU
  )

SET_TARGET_PROPERTIES(GetCUDADeviceProperties
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Utilities/DeviceManagement")
