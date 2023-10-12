#include "Visualization/OpenGLInterface/HandleGLError.h"
#include "Visualization/OpenGLInterface/OpenGLBufferObjectNames.h"
#include "gtest/gtest.h"

using Parameters =
  Visualization::OpenGLInterface::OpenGLBufferObjectNames::Parameters;
using Visualization::OpenGLInterface::HandleGLError;
using Visualization::OpenGLInterface::OpenGLBufferObjectNames;

namespace GoogleUnitTests
{
namespace Visualization
{
namespace OpenGLInterface
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(OpenGLBufferObjectNamesTests, ConstructibleWithDefaultParameters)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  OpenGLBufferObjectNames buffer_object {parameters};

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(OpenGLBufferObjectNamesTests, DestructsWithOneBufferObject)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  {
    OpenGLBufferObjectNames buffer_object {parameters};
  }

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(OpenGLBufferObjectNamesTests, ConstructibleWithTwoObjects)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  parameters.number_of_buffer_object_names_ = 2;

  OpenGLBufferObjectNames buffer_object {parameters};

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(OpenGLBufferObjectNamesTests, DestructibleWithTwoObjects)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  parameters.number_of_buffer_object_names_ = 2;

  {
    OpenGLBufferObjectNames buffer_object {parameters};
  }

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(OpenGLBufferObjectNamesTests, InitializeInitializesWithDefaultParameters)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  OpenGLBufferObjectNames buffer_object {parameters};

  // Segmentation Fault if we had defined the type to be GLuint* instead of
  // GLuint for the data member.
  EXPECT_TRUE(buffer_object.initialize());

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
  EXPECT_TRUE(gl_err.is_no_gl_error());
}

} // namespace OpenGLInterface
} // namespace Visualization
} // namespace GoogleUnitTests