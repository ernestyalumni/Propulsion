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

namespace GoogleUnitTests
{
namespace Visualization
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CUDAGraphicsResourceTests, DefaultConstructs)
{
  CUDAGraphicsResource cuda_graphics_resource {};

  EXPECT_FALSE(cuda_graphics_resource.is_registered());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CUDAGraphicsResourceTests, RegisterBufferObjectFailsWithoutGLUTWindow)
{
  HandleGLError gl_err {};
  Parameters parameters {};
  parameters.binding_target_ = GL_PIXEL_UNPACK_BUFFER_ARB;
  parameters.usage_ = GL_DYNAMIC_DRAW_ARB;

  OpenGLBufferObjectNames buffer_object {parameters};
  ASSERT_TRUE(buffer_object.initialize());
  CreateOpenGLBuffer create_buffer {};
  EXPECT_TRUE(create_buffer.create_buffer_object_data(parameters));

  CUDAGraphicsResource cuda_graphics_resource {};
  const CUDAGraphicsResource::Parameters cuda_parameters {};
  const auto handle_call = cuda_graphics_resource.register_buffer_object(
    cuda_parameters,
    buffer_object);

  EXPECT_FALSE(handle_call.is_cuda_success());

  EXPECT_FALSE(cuda_graphics_resource.is_registered());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CUDAGraphicsResourceTests, MapResourceWorksOnItsOwn)
{
  CUDAGraphicsResource cuda_graphics_resource {};
  const auto handle_map_resource = cuda_graphics_resource.map_resource();

  EXPECT_FALSE(handle_map_resource.is_cuda_success());
}

} // namespace Visualization
} // namespace GoogleUnitTests