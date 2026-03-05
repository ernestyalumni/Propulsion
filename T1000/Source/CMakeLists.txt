CMAKE_MINIMUM_REQUIRED (VERSION 3.14)

# Language of the project is CXX (for C++)
PROJECT (T1000 LANGUAGES CXX)

# BEFORE prepends the directories specified onto current list of directories.
# cf. https://cmake.org/cmake/help/v3.0/command/include_directories.html
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR})

# Include will load and run CMake code of this module, "CheckCXXCompilerFlag."
INCLUDE(CheckCXXCompilerFlag)

# cf. https://stackoverflow.com/questions/25451254/getting-cmake-check-cxx-compiler-flag-to-work

CHECK_CXX_COMPILER_FLAG(-std=c++20 COMPILER_SUPPORTS_CXX20)

# Check if Compiler supports the Working draft for ISO C++ 2020 standard.
CHECK_CXX_COMPILER_FLAG(-std=c++2a COMPILER_SUPPORTS_CXX2A)

IF(COMPILER_SUPPORTS_CXX20)
  SET(CMAKE_CXX_STANDARD "20") # C++20
  SET(CMAKE_CXX_STANDARD_REQUIRED ON) # ... is required
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
ELSEIF(COMPILER_SUPPORTS_CXX2A)
  SET(CMAKE_CXX_STANDARD "20") # C++20, but with c++2a
  SET(CMAKE_CXX_STANDARD_REQUIRED ON) # ... is required
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")
ELSE()
  SET(CMAKE_CXX_STANDARD "17")
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
ENDIF()

MESSAGE("CMAKE_CXX_STANDARD: " ${CMAKE_CXX_STANDARD})

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

# CoolProp: use pre-built static library (e.g. from
# ThirdParty/CoolProp/BuildGcc)
set(COOLPROP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../ThirdParty/CoolProp" CACHE PATH "CoolProp root")

# Check CoolProp root and library; inform or warn
if(EXISTS "${COOLPROP_ROOT}")
  message(STATUS "CoolProp root found: ${COOLPROP_ROOT}")
else()
  message(WARNING "CoolProp root not found: ${COOLPROP_ROOT}")
endif()

set(COOLPROP_LIB "${COOLPROP_ROOT}/BuildGcc/libCoolProp.a")
set(COOLPROP_INCLUDE "${COOLPROP_ROOT}/include")
set(COOLPROP_FMT_INCLUDE "${COOLPROP_ROOT}/externals/fmtlib/include")
if(NOT EXISTS "${COOLPROP_LIB}")
  message(FATAL_ERROR "CoolProp library not found: ${COOLPROP_LIB}")
endif()
add_library(CoolProp STATIC IMPORTED GLOBAL)
set_target_properties(CoolProp PROPERTIES
  IMPORTED_LOCATION "${COOLPROP_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${COOLPROP_INCLUDE};${COOLPROP_FMT_INCLUDE}"
)

#ADD_SUBDIRECTORY("${CMAKE_CURRENT_SOURCE_DIR}/../../../ThirdParty/CoolProp" CoolProp)

# Add_subdirectory call makes use of libraries.
# Add_subdirectory adds the source CMakeLists.txt and code files in the
# directory.

ADD_SUBDIRECTORY(UnitTests)


ADD_EXECUTABLE(Main Main.cpp)

## Manually change this to include CoolProp wherever you've placed it.
TARGET_LINK_LIBRARIES(Main CoolProp)

