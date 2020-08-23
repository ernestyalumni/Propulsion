# Dependencies
#
# Find a boost install with the libraries unit_test_framework.
# Boost filesystem needs boost system.
#
FIND_PACKAGE(Boost COMPONENTS filesystem system unit_test_framework REQUIRED)

# Build unit tests
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

ADD_EXECUTABLE(Check
  Main.cpp
  CoolPropTests/PropsSI_tests.cpp)

TARGET_LINK_LIBRARIES(Check
  CoolProp
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

#
# cf. https://cmake.org/cmake/help/v3.3/prop_tgt/RUNTIME_OUTPUT_DIRECTORY.html
# RUNTIME_OUTPUT_DIRECTORY - output directory in which to build RUNTIME target files.
# Property specifies directory into which runtime target files should be built.
# cf. https://cmake.org/cmake/help/v3.3/prop_tgt/RUNTIME_OUTPUT_DIRECTORY_CONFIG.html#prop_tgt:RUNTIME_OUTPUT_DIRECTORY_%3CCONFIG%3E
# Per-configuration output directory for RUNTIME target files.
# This is a per-configuration version of RUNTIME_OUTPUT_DIRECTORY target property.
#
SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")

ADD_EXECUTABLE(HelloWorldCoolProp
  HelloWorldCoolProp.cpp)

TARGET_LINK_LIBRARIES(HelloWorldCoolProp
  CoolProp)

SET_TARGET_PROPERTIES(HelloWorldCoolProp
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
