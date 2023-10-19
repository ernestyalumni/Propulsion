#ifndef INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_PARAMETERS_H
#define INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_PARAMETERS_H

#include <cstddef>

namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{

struct Parameters
{
  std::size_t width_dimension_;
  std::size_t height_dimension_;
  float c_real_;
  float c_imaginary_;
  int maximum_iterations_;
  float maximum_threshold_;
};

} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_PARAMETERS_H