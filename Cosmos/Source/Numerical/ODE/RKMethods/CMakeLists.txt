# Shared ODE target, usable without the Cosmos examples or T1000's CoolProp.
add_library(CosmosODE STATIC Coefficients/DOPRI5Coefficients.cpp)
target_include_directories(CosmosODE PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
target_compile_features(CosmosODE PUBLIC cxx_std_17)
