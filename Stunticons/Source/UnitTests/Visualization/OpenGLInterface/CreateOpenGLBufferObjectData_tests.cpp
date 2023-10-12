#include "Visualization/OpenGLInterface/CreateOpenGLBuffer.h"
#include "Visualization/OpenGLInterface/HandleGLError.h"
#include "Visualization/OpenGLInterface/OpenGLBufferObjectNames.h"
#include "gtest/gtest.h"

using Parameters =
  Visualization::OpenGLInterface::OpenGLBufferObjectNames::Parameters;
using Visualization::OpenGLInterface::CreateOpenGLBuffer;
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
TEST(CreateOpenGLBufferObjectDataTests, DefaultConstructs)
{
  HandleGLError gl_err {};

  Parameters parameters {};

  OpenGLBufferObjectNames buffer_object {parameters};

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

  OpenGLBufferObjectNames buffer_object {parameters};

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

  OpenGLBufferObjectNames buffer_object {parameters};
  ASSERT_TRUE(buffer_object.initialize());

  CreateOpenGLBuffer create_buffer {};

  EXPECT_TRUE(create_buffer.create_buffer_object_data(parameters));
}

} // namespace OpenGLInterface
} // namespace Visualization
} // namespace GoogleUnitTests