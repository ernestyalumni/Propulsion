SET(ALGEBRA_CU_SOURCES
  Modules/Vectors/VectorAddition.cu
  Modules/Vectors/DeviceVectorAdditionArrays.cu
  )

# TODO: This line doesn't seem to be needed.
SET_SOURCE_FILES_PROPERTIES(${ALGEBRA_CU_SOURCES} PROPERTIES LANGUAGE CUDA)

ADD_LIBRARY(ALGEBRA_CU
  ${ALGEBRA_CU_SOURCES}
  )

ADD_LIBRARY(ALGEBRA
  Modules/Vectors/HostVectorAdditionArrays.cpp
  )
