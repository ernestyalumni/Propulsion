#include "Visualization/OpenGLInterface/OpenGLBufferObjectParameters.h"
#include "gtest/gtest.h"

#include <GL/glext.h>

using Parameters = Visualization::OpenGLInterface::OpenGLBufferObjectParameters;

namespace GoogleUnitTests
{
namespace Visualization
{
namespace OpenGLInterface
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(OpenGLBufferObjectParametersTests, DefaultConstructs)
{
  Parameters parameters {};

  EXPECT_EQ(parameters.calculate_new_data_store_size(), 1048576);

  EXPECT_EQ(parameters.number_of_buffer_object_names_, 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(OpenGLBufferObjectParametersTests, Constructs)
{
  Parameters parameters {
    2,
    GL_PIXEL_UNPACK_BUFFER_ARB,
    GL_STREAM_DRAW_ARB,
    512,
    256};

  EXPECT_EQ(parameters.calculate_new_data_store_size(), 524288);
}

} // namespace OpenGLInterface
} // namespace Visualization
} // namespace GoogleUnitTests