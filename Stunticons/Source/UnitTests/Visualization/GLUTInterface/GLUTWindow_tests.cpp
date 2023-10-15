#include "Visualization/GLUTInterface/GLUTWindow.h"
#include "gtest/gtest.h"

#include <GL/glut.h> // GLUT_DOUBLE, GLUT_RGBA
#include <type_traits>

using namespace Visualization::GLUTInterface;

namespace GoogleUnitTests
{
namespace Visualization
{
namespace GLUTInterface
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GLUTWindowTests, DefaultConstructible)
{
  EXPECT_TRUE(std::is_default_constructible<GLUTWindow>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GLUTWindowTests, NoCopyConstructor)
{
  EXPECT_FALSE(std::is_copy_constructible<GLUTWindow>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GLUTWindowTests, NoMoveConstructor)
{
  EXPECT_FALSE(std::is_move_constructible<GLUTWindow>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GLUTWindowTests, NoCopyAssignment)
{
  EXPECT_FALSE(std::is_copy_assignable<GLUTWindow>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GLUTWindowTests, NoMoveAssignment)
{
  EXPECT_FALSE(std::is_move_assignable<GLUTWindow>());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GLUTWindowTests, InstanceConstructs)
{
  EXPECT_EQ(GLUTWindow::instance().get_window_identifier(), 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GLUTWindowTests, InitializeGlutInitializesWithParameters)
{
  GLUTWindow::Parameters parameters {{GLUT_DOUBLE | GLUT_RGBA}};
  parameters.display_name_ = "bitmap";
  parameters.width_ = 512;
  parameters.height_ = 256;

  // Trick GLUT into thinking we're passing command line arguments.
  int soo {1};
  char* foo = new char{};
  *foo = 'x';

  GLUTWindow::instance().initialize_glut(&soo, &foo, parameters);

  delete foo;

  // Typically the value is 1 but it's not always true.
  EXPECT_TRUE(GLUTWindow::instance().get_window_identifier() > 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GLUTWindowTests, ParametersConstructWithInputs)
{
  GLUTWindow::Parameters parameters {
    "bitmap",
    512,
    256,
    {GLUT_DOUBLE | GLUT_RGBA}};

  EXPECT_EQ(parameters.width_, 512);
  EXPECT_EQ(parameters.height_, 256);
  EXPECT_EQ(parameters.display_name_, "bitmap");
}

} // namespace GLUTInterface
} // namespace Visualization
} // namespace GoogleUnitTests