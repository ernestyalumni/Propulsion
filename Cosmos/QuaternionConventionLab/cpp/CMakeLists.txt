cmake_minimum_required(VERSION 3.20)
project(quaternion_convention_cpp LANGUAGES CXX)

add_executable(quaternion_convention main.cpp)
target_compile_features(quaternion_convention PRIVATE cxx_std_20)
target_compile_options(
  quaternion_convention
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)
