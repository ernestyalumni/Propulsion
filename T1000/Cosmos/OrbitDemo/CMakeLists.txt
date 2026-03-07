cmake_minimum_required(VERSION 3.16)
project(CosmosOrbitDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler flags ────────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# ── Include path ──────────────────────────────────────────────────────────────
# Headers live in T1000/Source/ (two levels up from this CMakeLists.txt)
# i.e. ../../Source relative to OrbitDemo/
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Source")

include_directories(
    "${SOURCE_DIR}"
)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(OrbitDemo main.cpp)

target_include_directories(OrbitDemo PRIVATE "${SOURCE_DIR}")

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS OrbitDemo DESTINATION bin)

# ── Build instructions ────────────────────────────────────────────────────────
# From T1000/Cosmos/OrbitDemo/:
#   mkdir build && cd build
#   cmake ..
#   cmake --build .
#   ./OrbitDemo
