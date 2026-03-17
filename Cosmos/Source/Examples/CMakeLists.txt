CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

#==============================================================================
# Astrodynamics Examples
#==============================================================================

#------------------------------------------------------------------------------
# Numerov Orbit Demo (C++ native executable)
# Uses Cosmos library headers only — no link dependency on Numerical lib.
#------------------------------------------------------------------------------
ADD_EXECUTABLE(NumerovOrbitDemo Astrodynamics/NumerovOrbitDemo.cpp)

TARGET_INCLUDE_DIRECTORIES(NumerovOrbitDemo PRIVATE
  ${CMAKE_SOURCE_DIR}
)

SET_TARGET_PROPERTIES(NumerovOrbitDemo PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
)

#------------------------------------------------------------------------------
# WASM Bridge (Emscripten)
# Builds wasm_bridge.cpp → numerov_orbit.js + numerov_orbit.wasm
#
# Usage from browser:
#   <script src="numerov_orbit.js"></script>
#   <script>
#     NumerovOrbitModule().then(Module => {
#       const create = Module.cwrap('numerov_orbit_create', 'number',
#                                   ['number', 'number', 'number']);
#       // ... see wasm_bridge.cpp for full API
#     });
#   </script>
#
# Build manually if Emscripten not in CMake toolchain:
#   emcc Astrodynamics/wasm_bridge.cpp -I ../../.. \
#     -o numerov_orbit.js -s WASM=1 -O3 -std=c++20 \
#     -s EXPORTED_FUNCTIONS='["_malloc","_free","_numerov_orbit_create","_numerov_orbit_destroy","_numerov_orbit_step","_numerov_orbit_step_n","_numerov_orbit_get_state","_numerov_orbit_get_earth_constants"]' \
#     -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","setValue","getValue","HEAPF64","HEAPU8"]' \
#     -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 \
#     -s EXPORT_NAME="NumerovOrbitModule"
#
# Note: setValue/getValue/HEAPF64/HEAPU8 must be in EXPORTED_RUNTIME_METHODS
#       (Emscripten 5+ requires explicit export of all heap/value helpers)
# Note: ENVIRONMENT flag omitted → supports both web and node environments
#------------------------------------------------------------------------------
IF(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  MESSAGE(STATUS "Emscripten toolchain detected — building WASM bridge")

  ADD_EXECUTABLE(numerov_orbit_wasm Astrodynamics/wasm_bridge.cpp)

  TARGET_INCLUDE_DIRECTORIES(numerov_orbit_wasm PRIVATE
    ${CMAKE_SOURCE_DIR}
  )

  SET_TARGET_PROPERTIES(numerov_orbit_wasm PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    LINK_FLAGS "-s WASM=1 \
                -s EXPORTED_FUNCTIONS='[\"_malloc\",\"_free\",\"_numerov_orbit_create\",\"_numerov_orbit_destroy\",\"_numerov_orbit_step\",\"_numerov_orbit_step_n\",\"_numerov_orbit_get_state\",\"_numerov_orbit_get_earth_constants\"]' \
                -s EXPORTED_RUNTIME_METHODS='[\"ccall\",\"cwrap\",\"setValue\",\"getValue\",\"HEAPF64\",\"HEAPU8\"]' \
                -s ALLOW_MEMORY_GROWTH=1 \
                -s MODULARIZE=1 \
                -s EXPORT_NAME=\"NumerovOrbitModule\" \
                -O3"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
    OUTPUT_NAME "numerov_orbit"
  )
ELSE()
  MESSAGE(STATUS "Native build — skipping WASM bridge (no Emscripten toolchain)")
  MESSAGE(STATUS "  To build WASM: source emsdk_env.sh && cmake .. -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")
ENDIF()

#------------------------------------------------------------------------------
# Install targets (optional)
#------------------------------------------------------------------------------
INSTALL(TARGETS NumerovOrbitDemo
  RUNTIME DESTINATION bin
)