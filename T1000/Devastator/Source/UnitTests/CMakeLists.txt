ENABLE_TESTING()

ADD_EXECUTABLE(Check
  Algebra/Modules/Vectors/Vector3_tests.cpp
  Numerical/Interpolation/Interpolation_tests.cpp
  Numerical/ODE/ODEIntTestSetup.cpp
  Numerical/ODE/ODEInt_tests.cpp
  Numerical/ODE/Output_tests.cpp
  Numerical/ODE/RK4_tests.cpp
  Numerical/ODE/RKMethods/CalculateNewYAndError_tests.cpp
  Numerical/ODE/RKMethods/Coefficients/ACoefficients_tests.cpp
  Numerical/ODE/RKMethods/Coefficients/KCoefficients_tests.cpp
  Numerical/ODE/RKMethods/RungeKuttaMethod_tests.cpp
  Numerical/ODE/RhsVanDerPol_tests.cpp
  Numerical/ODE/RhsVan_tests.cpp
  Numerical/ODE/StepperBase_tests.cpp
  Numerical/ODE/StepperDopr5_tests.cpp
  Utilities/BinarySearch_tests.cpp
  )

TARGET_LINK_LIBRARIES(Check
  Numerical
  Utilities
  gtest_main
  )

INCLUDE(GoogleTest)
gtest_discover_tests(Check)

SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
