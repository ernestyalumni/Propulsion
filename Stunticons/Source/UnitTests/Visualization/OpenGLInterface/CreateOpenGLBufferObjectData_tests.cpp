#include "Visualization/OpenGLInterface/CreateOpenGLBuffer.h"
#include "Visualization/OpenGLInterface/HandleGLError.h"
#include "Visualization/OpenGLInterface/BufferObjectNames.h"
#include "gtest/gtest.h"

using Parameters =
  Visualization::OpenGLInterface::BufferObjectNames::Parameters;
using Visualization::OpenGLInterface::CreateOpenGLBuffer;
using Visualization::OpenGLInterface::HandleGLError;
using Visualization::OpenGLInterface::BufferObjectNames;

namespace GoogleUnitTests
{
namespace Visualization
{
namespace OpenGLInterface
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CreateOpenGLBufferObjectDataTests, DefaultConstructs)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  BufferObjectNames buffer_object {parameters};

  CreateOpenGLBuffer create_buffer {};

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CreateOpenGLBufferObjectDataTests, ConstructsWithParameters)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  BufferObjectNames buffer_object {parameters};

  CreateOpenGLBuffer create_buffer {parameters};

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  CreateOpenGLBufferObjectDataTests,
  CreateBufferObjectDataWorksOnSingleObject)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  BufferObjectNames buffer_object {parameters};
  ASSERT_TRUE(buffer_object.initialize());

  CreateOpenGLBuffer create_buffer {};

  EXPECT_TRUE(create_buffer.create_buffer_object_data(parameters));
  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
}

//------------------------------------------------------------------------------
/// \url https://docs.gl/gl4/glBindBuffer
/// \details From Examples: Load a vertex buffer into OpenGL for later
/// rendering.
//------------------------------------------------------------------------------
TEST(CreateOpenGLBufferObjectDataTests, LoadVertexBuffer)
{
  HandleGLError gl_err {};

  Parameters parameters {};
  parameters.usage_ = GL_STATIC_DRAW;

  BufferObjectNames buffer_object {parameters};
  ASSERT_TRUE(buffer_object.initialize());

  CreateOpenGLBuffer create_buffer {};

  EXPECT_TRUE(create_buffer.create_buffer_object_data(parameters));
  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

//------------------------------------------------------------------------------
/// \url https://docs.gl/gl4/glBindBuffer
/// \details From Examples: Load an index buffer into OpenGL for later
/// rendering.
//------------------------------------------------------------------------------
TEST(CreateOpenGLBufferObjectDataTests, LoadIndexBuffer)
{
  HandleGLError gl_err {};

  Parameters parameters {};
  parameters.binding_target_ = GL_ELEMENT_ARRAY_BUFFER;
  parameters.usage_ = GL_STATIC_DRAW;

  BufferObjectNames buffer_object {parameters};

  ASSERT_TRUE(buffer_object.initialize());

  CreateOpenGLBuffer create_buffer {};

  EXPECT_TRUE(create_buffer.create_buffer_object_data(parameters));
  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

} // namespace OpenGLInterface
} // namespace Visualization
} // namespace GoogleUnitTests