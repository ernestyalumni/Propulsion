#ifndef INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_SIMPLE_CUDA2GL_H
#define INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_SIMPLE_CUDA2GL_H

namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{

void make_striped_pattern(
  const dim3 threads,
  const dim3 blocks,
  unsigned int* data,
  const int image_width);

} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_SIMPLE_CUDA2GL_H