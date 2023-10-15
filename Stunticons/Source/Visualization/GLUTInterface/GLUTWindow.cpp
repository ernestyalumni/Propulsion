#include "Visualization/GLUTInterface/GLUTWindow.h"

#include <GL/glut.h> // glutInit
#include <algorithm> // all_of
#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <numeric> // accumulate
#include <stdexcept>
#include <vector>

using std::size_t;

namespace Visualization
{
namespace GLUTInterface
{

std::array<unsigned int, 12> GLUTWindow::Parameters::valid_modes_ {
  GLUT_RGBA,
  GLUT_RGB,
  GLUT_INDEX,
  GLUT_SINGLE,
  GLUT_DOUBLE,
  GLUT_ACCUM,
  GLUT_ALPHA,
  GLUT_DEPTH,
  GLUT_STENCIL,
  GLUT_MULTISAMPLE,
  GLUT_STEREO,
  GLUT_LUMINANCE
};

void GLUTWindow::Parameters::set_modes(std::initializer_list<unsigned int> l)
{
  // https://en.cppreference.com/w/cpp/algorithm/all_any_none_of
  // Checks if unary predicate p returns true for all elements in range.
  // Returns true if unary predicate returns true for all elements in range.
  if (!std::all_of(
    l.begin(),
    l.end(),
    [](const unsigned int mode)
    {
      return std::find(valid_modes_.begin(), valid_modes_.end(), mode) !=
        valid_modes_.end();
    }))
  {
    throw std::invalid_argument("One or more values for modes are not valid!");
  }

  // https://en.cppreference.com/w/cpp/algorithm/accumulate
  // constexpr T accumulate(InputIt first, InputIt last, T init, BinaryOperation
  // op);
  // Initializes accumulator acc with initial value init and then modifies it
  // with acc = op(acc, *i) for every iterator i in range.
  // init - initial value of sum. op - binary operation function object that
  // will be applied.
  const unsigned int mode {
    std::accumulate(
      l.begin(),
      l.end(),
      0U,
      [](unsigned int a, unsigned int b)
      {
        return a | b;
      })};

  modes_ = mode;  
}

GLUTWindow::Parameters::Parameters(std::initializer_list<unsigned int> l):
  width_{0},
  height_{0}
{
  set_modes(l);
}

GLUTWindow::Parameters::Parameters(
  const std::string_view display_name,
  const size_t width,
  const size_t height,
  std::initializer_list<unsigned int> l
  ):
  modes_{0},
  display_name_{display_name},
  width_{width},
  height_{height}
{
  set_modes(l);
}

char* GLUTWindow::Parameters::get_display_name() const
{
  std::vector<char> buffer (display_name_.begin(), display_name_.end());
  buffer.push_back('\0');
  return &buffer[0];
}

//------------------------------------------------------------------------------
/// The following static class member variables *must* be initialized (and
/// initializes in the implementation, i.e. source, file since they are
/// non-const static data members and the compiler would otherwise complain) to
/// avoid linking problems because otherwise for the subsequent functions that
/// use them, they will treat the uninitialized variables as an error,
/// "undefined reference." In other words, you can't initialize them or have the
/// following lines of code *after* the code implementing the functions using
/// them.
//------------------------------------------------------------------------------

int GLUTWindow::window_identifier_ {0};

std::unique_ptr<GLUTWindow> GLUTWindow::instance_ {nullptr};

std::mutex GLUTWindow::mutex_;

void GLUTWindow::initialize_glut(
  int* argcp,
  char** argv,
  const Parameters& parameters)
{
  std::lock_guard<std::mutex> lock(mutex_);

  glutInit(argcp, argv);
  // https://www.opengl.org/resources/libraries/glut/spec3/node12.html
  glutInitDisplayMode(parameters.modes_);
  glutInitWindowSize(parameters.width_, parameters.height_);
  window_identifier_ = glutCreateWindow(parameters.get_display_name()); 

  // mutex_ is automatically released when lock goes out of scope.
}

GLUTWindow& GLUTWindow::instance()
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (instance_ == nullptr)
  {
    instance_ = std::make_unique<GLUTWindow>();
  }

  return *instance_;

  // mutex_ is automatically released when lock goes out of scope.
}

} // namespace GLUTInterface
} // namespace Visualization