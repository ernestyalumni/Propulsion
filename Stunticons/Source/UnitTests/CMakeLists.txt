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
  CUDA/Runtime_tests.cu
  DataStructures/Array_tests.cu
  Manifolds/Operators/FiniteDifference/C_Nu_Coefficients_tests.cu
  Manifolds/Operators/FiniteDifference/DirectionalDerivatives_tests.cu
  Utilities/DeviceManagement/GetCUDADeviceProperties_tests.cu
  Utilities/Testing/Manifolds/Operators/FiniteDifference/Stencil_tests.cu
  Visualization/float_to_color_tests.cu
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
  Visualization/CUDAGraphicsResource_tests.cu
  Visualization/GLUTInterface/GLUTWindow_tests.cpp
  Visualization/OpenGLInterface/BufferObjectNames_tests.cpp
  Visualization/OpenGLInterface/BufferObjectParameters_tests.cpp
  Visualization/OpenGLInterface/CreateOpenGLBufferObjectData_tests.cpp
  Visualization/OpenGLInterface/HandleGLError_tests.cpp
  )

TARGET_COMPILE_OPTIONS(Check PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --expt-extended-lambda
  >)

TARGET_LINK_LIBRARIES(Check
  ALGEBRA
  ALGEBRA_CU
  FIBERBUNDLES
  MANIFOLDS
  Utilities
  Utilities_CU
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