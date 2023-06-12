# Find Eigen
FIND_PACKAGE(Eigen3 QUIET)

IF (Eigen3_FOUND)
  MESSAGE(STATUS "Eigen3 found")
  ADD_EXECUTABLE(CheckPerformanceTests
    Algebra/Solvers/SetupBiconjugateGradientStabilizedWithEigen.cpp
    Algebra/Solvers/SetupBiconjugateGradientStabilizedWithEigen_tests.cpp
    )

  TARGET_LINK_LIBRARIES(CheckPerformanceTests
    ALGEBRA
    UTILITIES
    gtest_main
    )

  SET_TARGET_PROPERTIES(CheckPerformanceTests
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

  ADD_EXECUTABLE(CompareConjugateGradientMethodsCPU
    Algebra/Solvers/SetupBiconjugateGradientStabilizedWithEigen.cpp
    ConjugateGradientMethodsCPU.cpp
    )

  TARGET_LINK_LIBRARIES(CompareConjugateGradientMethodsCPU
    ALGEBRA
    UTILITIES
    gtest_main
    )

  SET_TARGET_PROPERTIES(CompareConjugateGradientMethodsCPU
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

ELSE()
  MESSAGE(STATUS "Eigen3 not found, not building related objects and executables")
ENDIF()

ADD_EXECUTABLE(CheckConjugateGradientMethodsPerformance
  ConjugateGradientMethodsPerformanceTest.cu
  )

TARGET_LINK_LIBRARIES(CheckConjugateGradientMethodsPerformance
  ALGEBRA
  ALGEBRA_CU
  UTILITIES
  UTILITIES_CU
  )

SET_TARGET_PROPERTIES(CheckConjugateGradientMethodsPerformance
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")