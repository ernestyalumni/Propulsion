#include "HandleGLError.h"

#include <charconv>
// Use #include <format> when GCC 13+ for C++20.
#include <string>

namespace Visualization
{
namespace OpenGLInterface
{

HandleGLError::HandleGLError():
  gl_error_{GL_NO_ERROR}
{}

std::string_view HandleGLError::operator()()
{
  // Check for error.
  gl_error_ = glGetError();

  if (gl_error_ != GL_NO_ERROR)
  {
    std::string return_result {
      // Use format when GCC 13+
      // std::format("GL Error in file: {} in line {}:\n", __FILE__, __LINE__)};
      "GL Error in file: "};

    // TODO: Consider using std::to_chars instead of std::to_string;
    return_result.append(__FILE__);
    return_result.append(" in line ");
    return_result.append(std::to_string(__LINE__));
    return_result.append(":\n");
    return_result.append(#gl_error_);

    return std::string_view {return_result};
  }

  return std::string_view {#gl_error_);
}

} // namespace OpenGLInterface
} // namespace Visualization
