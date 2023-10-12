ENABLE_TESTING()

SET(TEST_CU_SOURCES
  Algebra/Modules/Matrices/CompressedSparseRow_tests.cu
  Algebra/Modules/Morphisms/SparseMatrixMorphism_tests.cu
  Algebra/Modules/Vectors/Array_tests.cu
  Algebra/Modules/Vectors/CuBLASVectorOperations_tests.cu
  Algebra/Modules/Vectors/Vector2_tests.cu
  Algebra/Modules/Vectors/VectorAddition_tests.cu
  Algebra/Solvers/BiconjugateGradientStabilized_tests.cu
  Algebra/Solvers/ConjugateGradient_tests.cu
  Algebra/Solvers/SetupBiconjugateGradientStabilized_tests.cu
  Utilities/DeviceManagement/GetCUDADeviceProperties_tests.cu
  )

# TODO: This line doesn't seem to be needed.
SET_SOURCE_FILES_PROPERTIES(${TEST_CU_SOURCES} PROPERTIES LANGUAGE CUDA)

ADD_EXECUTABLE(Check
  ${TEST_CU_SOURCES}
  Algebra/Modules/Matrices/GenerateCompressedSparseRowMatrix_tests.cpp
  Algebra/Modules/Matrices/HostCompressedSparseRow_tests.cpp
  Algebra/Modules/Vectors/HostArrays_tests.cpp
  Algebra/Modules/Vectors/HostVectorAdditionArrays_tests.cpp
  Arithmetic/IntegerPower_tests.cpp
  FiberBundles/PgmGeometry/CellGrid_tests.cpp
  FiberBundles/PgmGeometry/Cell_tests.cpp
  Manifolds/Euclidean/PgmGeometry/Grid2d_tests.cpp
  Manifolds/Euclidean/PgmGeometry/initialize_lid_driven_cavity_tests.cpp
  Utilities/FileIO/FilePath_tests.cpp
  Utilities/FileIO/ReadMatrixMarketFile_tests.cpp
  Utilities/FileIO/TurbulentFlow/Configuration_tests.cpp
  Utilities/FileIO/TurbulentFlow/ParseGeometryFile_tests.cpp
  Utilities/FileIO/TurbulentFlow/ReadConfiguration_tests.cpp
  Visualization/OpenGLInterface/CreateOpenGLBufferObjectData_tests.cpp
  Visualization/OpenGLInterface/HandleGLError_tests.cpp
  Visualization/OpenGLInterface/OpenGLBufferObjectNames_tests.cpp
  Visualization/OpenGLInterface/OpenGLBufferObjectParameters_tests.cpp
  )

TARGET_LINK_LIBRARIES(Check
  ALGEBRA
  ALGEBRA_CU
  FIBERBUNDLES
  MANIFOLDS
  UTILITIES
  UTILITIES_CU
  Visualization
  gtest_main
  ${OPENGL_LIBRARIES}
  )

INCLUDE(GoogleTest)
gtest_discover_tests(Check)

# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_SEPARABLE_COMPILATION.html
# CUDA_SEPARABLE_COMPILATION enables separate compilation for all CUDA files
# for the given target.
# It should allow us to compile multiple CUDA source files into separate device
# object files and then link them together into a single executable or library.
# It should be necessary when using device code linking (e.g. __device__ or
# __global__ functions in different translation units).
SET_TARGET_PROPERTIES(Check
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")