#ifndef INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_PARAMETERS_H
#define INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_PARAMETERS_H

namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{
namespace JuliaSet
{

struct Parameters
{
  // Use int because the width, and height are involved in calculations with
  // ints.
  int width_dimension_;
  int height_dimension_;

  float c_real_;
  float c_imaginary_;
  // Originally 200; this parameter tests what points go to infinity. Higher
  // value makes the set look "lacy."
  int maximum_iterations_;
  // Magnitude threshold that determines if point is in Julia set.
  float magnitude_threshold_;
};

inline Parameters get_default_julia_parameters(
  const int width_dimension,
  const int height_dimension)
{
  return Parameters{
    width_dimension,
    height_dimension,
    -0.8168,
    0.1583,
    300,
    1000};
}

} // namespace JuliaSet
} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_PARAMETERS_H