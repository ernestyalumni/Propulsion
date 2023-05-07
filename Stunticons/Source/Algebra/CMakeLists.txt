SET(ALGEBRA_CU_SOURCES
  Modules/Matrices/CompressedSparseRow.cu
  Modules/Vectors/VectorAddition.cu
  Modules/Vectors/DeviceVectorAdditionArrays.cu
  )

# TODO: This line doesn't seem to be needed.
SET_SOURCE_FILES_PROPERTIES(${ALGEBRA_CU_SOURCES} PROPERTIES LANGUAGE CUDA)

ADD_LIBRARY(ALGEBRA_CU
  ${ALGEBRA_CU_SOURCES}
  )

TARGET_LINK_LIBRARIES(ALGEBRA_CU ${CUDA_cusparse_LIBRARY})

ADD_LIBRARY(ALGEBRA
  Modules/Matrices/GenerateCompressedSparseRowMatrix.cpp
  Modules/Matrices/HostCompressedSparseRow.cpp
  Modules/Vectors/HostVectorAdditionArrays.cpp
  )
