#include "StripedPattern.h"
#include "Visualization/GLUTInterface/GLUTWindow.h"
#include "Visualization/OpenGLInterface/HandleGLError.h"

#include <cuda_runtime.h>
#include <GL/glut.h> // GLUT_RGBA

using GLUTWindowParameters =
  Visualization::GLUTInterface::GLUTWindow::Parameters;
using Visualization::OpenGLInterface::HandleGLError;

namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{

StripedPattern::Parameters::Parameters(
  const unsigned int image_width,
  const unsigned int image_height,
  const dim3 blocks
  ):
  image_width_{image_width},
  image_height_{image_height},
  blocks_{blocks},
  threads_{image_width / blocks.x, image_height / blocks.y, 1}
{}

const StripedPattern::Parameters StripedPattern::default_parameters_{
  512,
  512,
  dim3{16, 16, 1}};

const GLUTWindowParameters
  StripedPattern::default_glut_window_parameters_{
    "CUDA OpenGL post-processing",
    512,
    512,
    {GLUT_RGBA, GLUT_ALPHA, GLUT_DOUBLE, GLUT_DEPTH}};

void StripedPattern::run(int* argcp, char** argv)
{
  // Start of "initGL"

  ::Visualization::GLUTInterface::GLUTWindow::instance().initialize_glut(
    argcp,
    argv,
    default_glut_window_parameters_);

  ::Visualization::GLUTInterface::GLUTWindow::clear_color_buffers(
    0.5,
    0.5,
    0.5,
    1.0);

  // viewport
  glViewport(0, 0, 512, 512);

  // projection
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (GLfloat)512 / (GLfloat)512, 0.1f,
                 10.0f);

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  glEnable(GL_LIGHT0);
  float red[] = {1.0f, 0.1f, 0.1f, 1.0f};
  float white[] = {1.0f, 1.0f, 1.0f, 1.0f};
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
  glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0f);

  HandleGLError gl_err {};

  // End of "initGL"

}

} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests