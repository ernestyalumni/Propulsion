#ifndef INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_STRIPED_PATTERN_H
#define INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_STRIPED_PATTERN_H

#include "Visualization/GLUTInterface/GLUTWindow.h"

#include <cuda_runtime.h>

namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{

class StripedPattern
{
  public:

    using GLUTWindowParameters =
      ::Visualization::GLUTInterface::GLUTWindow::Parameters;

    struct Parameters
    {
      Parameters(
        const unsigned int image_width,
        const unsigned int image_height,
        const dim3 blocks);

      unsigned int image_width_;
      unsigned int image_height_;
      dim3 blocks_;
      dim3 threads_;
    };

    static const Parameters default_parameters_;

    static const GLUTWindowParameters default_glut_window_parameters_;

    void run(int* argcp, char** argv);
};

} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_STRIPED_PATTERN_H