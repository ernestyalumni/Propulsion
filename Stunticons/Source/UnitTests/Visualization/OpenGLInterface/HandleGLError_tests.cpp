#include "Visualization/OpenGLInterface/HandleGLError.h"
#include "gtest/gtest.h"

#include <atomic> // std::memory_order, for examples.
#include <string_view>

using Visualization::OpenGLInterface::HandleGLError;
using std::memory_order;
using std::string_view;

namespace GoogleUnitTests
{
namespace Visualization
{
namespace OpenGLInterface
{

string_view stringizing_example(const memory_order memory_access)
{
  // Compiler will complain about this syntax because # is considered a stray
  // #.
  // const string_view memory_access_as_str {#memory_access};

  #define RETURN_AS_STRING_MACRO(arg) #arg

  const string_view memory_access_as_str {
    RETURN_AS_STRING_MACRO(memory_access)};

  return memory_access_as_str;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HandleGLErrorTests, StringizerReturnsArgumentNameAsIs)
{
  string_view result {stringizing_example(memory_order::relaxed)};

  EXPECT_EQ(result, "memory_access");

  result = stringizing_example(memory_order::consume);

  EXPECT_EQ(result, "memory_access");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HandleGLErrorTests, DefaultConstructs)
{
  HandleGLError gl_err {};

  EXPECT_TRUE(gl_err.is_no_gl_error());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HandleGLErrorTests, CallOperatorWorks)
{
  HandleGLError gl_err {};

  EXPECT_EQ(gl_err(), "GL_NO_ERROR");
}

} // namespace OpenGLInterface
} // namespace Visualization
} // namespace GoogleUnitTests