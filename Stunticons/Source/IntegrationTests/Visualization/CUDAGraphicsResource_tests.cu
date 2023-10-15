#include "Visualization/CUDAGraphicsResource.h"
#include "Visualization/GLUTInterface/GLUTWindow.h"
#include "Visualization/OpenGLInterface/CreateOpenGLBuffer.h"
#include "Visualization/OpenGLInterface/HandleGLError.h"
#include "Visualization/OpenGLInterface/OpenGLBufferObjectNames.h"
#include "gtest/gtest.h"

#include <GL/glut.h> // GLUT_DOUBLE, GLUT_RGBA

using Parameters =
  Visualization::OpenGLInterface::OpenGLBufferObjectNames::Parameters;
using Visualization::CUDAGraphicsResource;
using Visualization::OpenGLInterface::CreateOpenGLBuffer;
using Visualization::OpenGLInterface::HandleGLError;
using Visualization::OpenGLInterface::OpenGLBufferObjectNames;

namespace IntegrationTests
{
namespace Visualization
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CUDAGraphicsResourceTests, RegistersBufferObjectWithGLUTWindow)
{
  HandleGLError gl_err {};

  ::Visualization::GLUTInterface::GLUTWindow::Parameters glut_parameters {
    "bitmap",
    512,
    256,
    {GLUT_DOUBLE | GLUT_RGBA}};

  // Trick GLUT into thinking we're passing command line arguments.
  int soo {1};
  char* foo = new char{};
  *foo = 'x';

  ::Visualization::GLUTInterface::GLUTWindow::instance().initialize_glut(
    &soo,
    &foo,
    glut_parameters);

  Parameters parameters {};
  parameters.binding_target_ = GL_PIXEL_UNPACK_BUFFER_ARB;
  parameters.usage_ = GL_DYNAMIC_DRAW_ARB;

  OpenGLBufferObjectNames buffer_object {parameters};
  ASSERT_TRUE(buffer_object.initialize());
  CreateOpenGLBuffer create_buffer {};
  ASSERT_TRUE(create_buffer.create_buffer_object_data(parameters));

  CUDAGraphicsResource cuda_graphics_resource {};
  const CUDAGraphicsResource::Parameters cuda_parameters {};
  const auto handle_call = cuda_graphics_resource.register_buffer_object(
    cuda_parameters,
    buffer_object);

  delete foo;

  EXPECT_TRUE(handle_call.is_cuda_success());
  EXPECT_TRUE(cuda_graphics_resource.is_registered());

}

} // namespace Visualization
} // namespace IntegrationTests