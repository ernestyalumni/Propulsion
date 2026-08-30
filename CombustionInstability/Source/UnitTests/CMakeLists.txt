ENABLE_TESTING()

SET(TEST_CU_SOURCES
  SpectralAnalysis/PowerSpectralDensity_tests.cu
  SpectralAnalysis/InstabilityDetector_tests.cu
  )

ADD_EXECUTABLE(CheckCombustionInstability
  ${TEST_CU_SOURCES}
  )

TARGET_LINK_LIBRARIES(CheckCombustionInstability
  SpectralAnalysisCU
  CUDA::cufft
  gmock_main
  gtest_main
  )

INCLUDE(GoogleTest)
gtest_discover_tests(CheckCombustionInstability)

SET_TARGET_PROPERTIES(CheckCombustionInstability
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET_TARGET_PROPERTIES(CheckCombustionInstability
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
