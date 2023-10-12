#include "HandleGLError.h"

#include <charconv>
// Use #include <format> when GCC 13+ for C++20.
#include <string>
#include <string_view>

using std::string_view;

namespace Visualization
{
namespace OpenGLInterface
{

string_view HandleGLError::gl_error_to_string(const GLenum error)
{
  // Defining the macro in this manner as a switch case is necessary because the
  // stringizer macro is effectively literal in how it converts an argument into
  // a string.
  #define RETURN_STRING_BY_CASE_MACRO(arg) case arg: return #arg

  switch(error)
  {
    RETURN_STRING_BY_CASE_MACRO(GL_NO_ERROR);

    // https://www.khronos.org/opengl/wiki/OpenGL_Error
    RETURN_STRING_BY_CASE_MACRO(GL_INVALID_ENUM);
    RETURN_STRING_BY_CASE_MACRO(GL_INVALID_VALUE);
    RETURN_STRING_BY_CASE_MACRO(GL_INVALID_OPERATION);
    RETURN_STRING_BY_CASE_MACRO(GL_STACK_OVERFLOW);
    RETURN_STRING_BY_CASE_MACRO(GL_STACK_UNDERFLOW);
    RETURN_STRING_BY_CASE_MACRO(GL_OUT_OF_MEMORY);
    RETURN_STRING_BY_CASE_MACRO(GL_INVALID_FRAMEBUFFER_OPERATION);
    RETURN_STRING_BY_CASE_MACRO(GL_CONTEXT_LOST);
    default: break;
  }

  return "UNKNOWN";
}

HandleGLError::HandleGLError():
  gl_error_{GL_NO_ERROR}
{}

string_view HandleGLError::operator()()
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
    return_result.append(gl_error_to_string(gl_error_));

    return std::string_view {return_result};
  }

  return gl_error_to_string(gl_error_);
}

} // namespace OpenGLInterface
} // namespace Visualization
