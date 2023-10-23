#ifndef INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SAMPLE_CODE_SAMPLE_CODE_H
#define INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SAMPLE_CODE_SAMPLE_CODE_H

#include "Visualization/CUDAGraphicsResource.h"
#include "Visualization/GLFWInterface/GLFWWindow.h"

#include <cuda_runtime.h>

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace SampleCode
{

class SampleCode
{
  public:

    using GLFWWindowParameters =
      ::Visualization::GLFWInterface::GLFWWindow::Parameters;

    static const GLFWWindowParameters default_glfw_window_parameters_;

    bool run(int* argcp, char** argv);

  private:


};

} // namespace SampleCode
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SAMPLE_CODE_SAMPLE_CODE_H