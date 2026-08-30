CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

MESSAGE("CMAKE_CUDA_ARCHITECTURES before SET: " ${CMAKE_CUDA_ARCHITECTURES})

IF(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Turing  SM75: RTX 2080, T4
  # Ampere  SM86: RTX 3060, RTX 3070, RTX 3080
  # Ada     SM89: RTX 4090
  SET(CMAKE_CUDA_ARCHITECTURES 75 86 89)
ELSE()
  SET(CMAKE_CUDA_ARCHITECTURES 75 86 89)
ENDIF()

MESSAGE("CMAKE_CUDA_ARCHITECTURES after SET: " ${CMAKE_CUDA_ARCHITECTURES})

PROJECT(CombustionInstability LANGUAGES CXX CUDA)

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR})

INCLUDE(CheckCXXCompilerFlag)

################################################################################
## Compiler flags
################################################################################

CHECK_CXX_COMPILER_FLAG(-std=c++20 COMPILER_SUPPORTS_CXX20)
CHECK_CXX_COMPILER_FLAG(-std=c++2a COMPILER_SUPPORTS_CXX2A)

IF(COMPILER_SUPPORTS_CXX20)
  SET(CMAKE_CXX_STANDARD "20")
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
ELSEIF(COMPILER_SUPPORTS_CXX2A)
  SET(CMAKE_CXX_STANDARD "20")
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")
ELSE()
  SET(CMAKE_CXX_STANDARD "17")
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
ENDIF()

################################################################################
## CUDA toolkit (for cuFFT)
################################################################################

FIND_PACKAGE(CUDAToolkit REQUIRED)

MESSAGE("CUDAToolkit_VERSION: " ${CUDAToolkit_VERSION})

IF(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
  SET(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
  SET(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
ENDIF()

IF(NOT (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0) AND COMPILER_SUPPORTS_CXX20)
  SET(CMAKE_CUDA_STANDARD 20)
  SET(CMAKE_CUDA_FLAGS "-std=c++20")
ENDIF()

################################################################################
## ccache
################################################################################

FIND_PROGRAM(CCACHE_FOUND ccache)
IF(CCACHE_FOUND)
  SET(CMAKE_CXX_COMPILER_LAUNCHER ccache)
ENDIF()

################################################################################
## GoogleTest
################################################################################

INCLUDE(FetchContent)

IF(CMAKE_VERSION VERSION_LESS "3.23.0")
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
ELSE()
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    DOWNLOAD_EXTRACT_TIMESTAMP true
  )
ENDIF()

FetchContent_MakeAvailable(googletest)

ADD_SUBDIRECTORY(SpectralAnalysis)
ADD_SUBDIRECTORY(UnitTests)
ADD_SUBDIRECTORY(PerformanceTests)
