#include "Visualization/OpenGLInterface/OpenGLBufferObjectNames.h"
#include "gtest/gtest.h"

using Parameters =
  Visualization::OpenGLInterface::OpenGLBufferObjectNames::Parameters;
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
  Parameters parameters {};

  OpenGLBufferObjectNames buffer_object {parameters};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(OpenGLBufferObjectNamesTests, DestructsWithOneBufferObject)
{
  Parameters parameters {};

  {
    OpenGLBufferObjectNames buffer_object {parameters};
  }

  SUCCEED();
}

} // namespace OpenGLInterface
} // namespace Visualization
} // namespace GoogleUnitTests