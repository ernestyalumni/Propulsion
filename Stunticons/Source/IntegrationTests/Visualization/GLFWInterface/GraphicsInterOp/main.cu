#include "IntegrationTests/Visualization/GLFWInterface/GraphicsInterOp/CalculatePixelValues.h"
#include "Visualization/GLFWInterface/MakeBitMap.h"

#include <cstddef>

using
  IntegrationTests::Visualization::GLFWInterface::GraphicsInterOp::
    CalculatePixelValues;

using Visualization::GLFWInterface::MakeBitMap;

int main(int argc, char* argv[])
{
  constexpr std::size_t dimension {1024};

  const auto buffer_object_parameters =
    MakeBitMap::make_default_buffer_object_parameters(dimension, dimension);

  const auto draw_pixels_parameters =
    MakeBitMap::make_default_draw_pixels_parameters(dimension, dimension);

  CalculatePixelValues calculate_pixel_values {
    CalculatePixelValues::Parameters {dimension, 16}};

  const MakeBitMap::GLFWWindowParameters glfw_window_parameters {
    "Graphics InterOp example",
    dimension,
    dimension};

  MakeBitMap make_bit_map {
    glfw_window_parameters,
    buffer_object_parameters,
    MakeBitMap::CUDAGraphicsResource::Parameters {},
    draw_pixels_parameters};

  return make_bit_map.run(calculate_pixel_values);
}