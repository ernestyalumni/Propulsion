#ifndef INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_IS_IN_JULIA_SET_H
#define INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_IS_IN_JULIA_SET_H

#include "IntegrationTests/Visualization/GLUTInterface/JuliaSet/Parameters.h"

#include <cstddef>

namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{

class IsInJuliaSet
{
  public:

    IsInJuliaSet(
      const std::size_t width_dimension,
      const std::size_t height_dimension,
      const float c_real,
      const float c_imaginary,
      const int maximum_iterations,
      const float maximum_threshold
      ):
      width_dimension_{width_dimension},
      height_dimension_{height_dimension},
      c_real_{c_real},
      c_imaginary_{c_imaginary},
      maximum_iterations_{maximum_iterations},
      maximum_threshold_{maximum_threshold}
    {}

    IsInJuliaSet(const Parameters& parameters):
      IsInJuliaSet{
        parameters.width_dimension_,
        parameters.height_dimension_,
        parameters.c_real_,
        parameters.c_imaginary_,
        parameters.maximum_iterations_,
        parameters.maximum_threshold_}
    {}

    __device__ int is_in_julia_set(const int x, const int y, const float scale);

    const std::size_t width_dimension_;
    const std::size_t height_dimension_;
    const float c_real_;
    const float c_imaginary_;
    const int maximum_iterations_;
    const float maximum_threshold_;
};

__global__ void is_in_julia_set(
  uchar4* ptr,
  const float scale,
  const Parameters& parameters);

} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_IS_IN_JULIA_SET_H