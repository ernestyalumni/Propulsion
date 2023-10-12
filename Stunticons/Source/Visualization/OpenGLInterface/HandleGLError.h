#ifndef VISUALIZATION_OPENGL_INTERFACE_HANDLE_GL_ERROR_H
#define VISUALIZATION_OPENGL_INTERFACE_HANDLE_GL_ERROR_H

#include <GL/gl.h>
#include <string_view>

namespace Visualization
{
namespace OpenGLInterface
{

class HandleGLError
{
  public:

    //--------------------------------------------------------------------------
    /// \ref helper_gl.h in cuda-samples, Common/
    /// \ref https://www.khronos.org/opengl/wiki/OpenGL_Error
    //--------------------------------------------------------------------------
    static std::string_view gl_error_to_string(const GLenum error);

    HandleGLError();

    ~HandleGLError() = default;

    std::string_view operator()();

    inline bool is_no_gl_error() const
    {
      return gl_error_ == GL_NO_ERROR;
    }

  private:

    GLenum gl_error_;
};

} // namespace OpenGLInterface
} // namespace Visualization

#endif // VISUALIZATION_OPENGL_INTERFACE_HANDLE_GL_ERROR_H