CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

# Astrodynamics Examples - standalone executables
# These examples are self-contained and don't require linking to Cosmos libraries

# Numerov Orbit Demo
ADD_EXECUTABLE(NumerovOrbitDemo Astrodynamics/NumerovOrbitDemo.cpp)
TARGET_INCLUDE_DIRECTORIES(NumerovOrbitDemo PRIVATE ${CMAKE_SOURCE_DIR})

# Set C++ standard
SET_TARGET_PROPERTIES(NumerovOrbitDemo PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
