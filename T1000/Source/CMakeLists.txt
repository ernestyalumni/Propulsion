CMAKE_MINIMUM_REQUIRED (VERSION 3.14)

# Language of the project is CXX (for C++)
PROJECT (T1000 LANGUAGES CXX)

# BEFORE prepends the directories specified onto current list of directories.
# cf. https://cmake.org/cmake/help/v3.0/command/include_directories.html
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR})

# Include will load and run CMake code of this module, "CheckCXXCompilerFlag."
INCLUDE(CheckCXXCompilerFlag)

# cf. https://stackoverflow.com/questions/25451254/getting-cmake-check-cxx-compiler-flag-to-work
CHECK_CXX_COMPILER_FLAG(-std=c++17 COMPILER_SUPPORTS_CXX17)

IF(COMPILER_SUPPORTS_CXX20)
  SET(CMAKE_CXX_STANDARD 20) # C++20
  SET(CMAKE_CXX_STANDARD_REQUIRED ON) # ... is required
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
ELSE()
  SET(CMAKE_CXX_STANDARD 17)
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
ENDIF()

FIND_PROGRAM(CCACHE_FOUND ccache)

IF(CCACHE_FOUND)
  SET(CMAKE_CXX_COMPILER_LAUNCHER ccache)
ENDIF()

SET(BOOST_MIN_VERSION "1.61.0")
FIND_PACKAGE(Boost)

# Include directories so that header files can be found.
INCLUDE_DIRECTORIES(BOOST_INCLUDES ${Boost_INCLUDE_DIRS})

# CoolProp
## cf. http://www.coolprop.org/coolprop/wrappers/StaticLibrary/index.html

SET(COOLPROP_STATIC_LIBRARY true)

ADD_SUBDIRECTORY("${CMAKE_CURRENT_SOURCE_DIR}/../../../CoolProp" CoolProp)

ADD_EXECUTABLE(Main Main.cpp)

## Manually change this to include CoolProp wherever you've placed it.
TARGET_LINK_LIBRARIES(Main CoolProp)

