FIND_PACKAGE(Boost COMPONENTS filesystem system unit_test_framework REQUIRED)

ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

ADD_EXECUTABLE(CheckOrbitalMechanics
  test_main.cpp
  test_TwoBody.cpp
  test_OrbitalElements.cpp
  test_Propagator.cpp)

TARGET_INCLUDE_DIRECTORIES(CheckOrbitalMechanics PRIVATE ${CMAKE_SOURCE_DIR})

TARGET_LINK_LIBRARIES(CheckOrbitalMechanics
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

SET_TARGET_PROPERTIES(CheckOrbitalMechanics PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
