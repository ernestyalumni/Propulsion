#include "Visualization/OpenGLInterface/HandleGLError.h"
#include "Visualization/OpenGLInterface/BufferObjectNames.h"
#include "gtest/gtest.h"

using Parameters =
  Visualization::OpenGLInterface::BufferObjectNames::Parameters;
using Visualization::OpenGLInterface::BufferObjectNames;
using Visualization::OpenGLInterface::HandleGLError;

namespace GoogleUnitTests
{
namespace Visualization
{
namespace OpenGLInterface
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BufferObjectNamesTests, ConstructibleWithDefaultParameters)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  BufferObjectNames buffer_object {parameters};

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BufferObjectNamesTests, DestructsWithOneBufferObject)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  {
    BufferObjectNames buffer_object {parameters};
  }

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BufferObjectNamesTests, ConstructibleWithTwoObjects)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  parameters.number_of_buffer_object_names_ = 2;

  BufferObjectNames buffer_object {parameters};

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BufferObjectNamesTests, DestructibleWithTwoObjects)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  parameters.number_of_buffer_object_names_ = 2;

  {
    BufferObjectNames buffer_object {parameters};
  }

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BufferObjectNamesTests, InitializeInitializesWithDefaultParameters)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  BufferObjectNames buffer_object {parameters};

  // Segmentation Fault if we had defined the type to be GLuint* instead of
  // GLuint for the data member.
  EXPECT_TRUE(buffer_object.initialize());

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

//------------------------------------------------------------------------------
/// \url https://docs.gl/gl4/glBindBuffer
/// \details From Examples: Load an index buffer into OpenGL for later
/// rendering.
//------------------------------------------------------------------------------
TEST(BufferObjectNamesTests, LoadIndexBuffer)
{
  HandleGLError gl_err {};

  Parameters parameters {};
  parameters.binding_target_ = GL_ELEMENT_ARRAY_BUFFER;

  BufferObjectNames buffer_object {parameters};

  EXPECT_TRUE(buffer_object.initialize());

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

//------------------------------------------------------------------------------
/// \url https://docs.gl/gl4/glBindBuffer
/// \details From Examples: Render an indexed buffer object using texture UV and
/// normal vertex attributes.
//------------------------------------------------------------------------------
TEST(BufferObjectNamesTests, VertexAndTextureUV)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  BufferObjectNames vertex_buffer_object {parameters};

  parameters.binding_target_ = GL_ELEMENT_ARRAY_BUFFER;

  BufferObjectNames index_buffer_object {parameters};

  EXPECT_TRUE(vertex_buffer_object.initialize());
  EXPECT_TRUE(index_buffer_object.initialize());

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

} // namespace OpenGLInterface
} // namespace Visualization
} // namespace GoogleUnitTests