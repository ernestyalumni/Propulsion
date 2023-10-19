#ifndef VISUALIZATION_OPENGL_INTERFACE_DRAW_PIXELS_H
#define VISUALIZATION_OPENGL_INTERFACE_DRAW_PIXELS_H

#include "Visualization/OpenGLInterface/HandleGLError.h"

#include <GL/glut.h>
#include <array>
#include <cstddef>

namespace Visualization
{
namespace OpenGLInterface
{

class DrawPixels
{
  public:

    using HandleGLError = Visualization::OpenGLInterface::HandleGLError;

    struct Parameters
    {
      static std::array<GLenum, 13> valid_format_;
      static std::array<GLenum, 21> valid_type_;

      Parameters(
        const std::size_t width,
        const std::size_t height,
        const GLenum format,
        const GLenum type
        ):
        width_{width},
        height_{height},
        format_{format},
        type_{type}
      {}

      std::size_t width_;
      std::size_t height_;
      GLenum format_;
      GLenum type_;
    };

    DrawPixels() = default;

    //--------------------------------------------------------------------------
    /// https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glDrawPixels.xml
    /// void glDrawPixels(
    ///   GLsizei width,
    ///   GLsizei height,
    ///   GLenum format,
    ///   GLenum type,
    ///   const void* data);
    /// data - specifies a pointer to pixel data.
    /// Data is read from data as a sequence of signed or unsigned bytes, signed
    /// or unsigned shorts, signed or unsigned integers, or single-precision
    /// floating-point values, depending on type. When type is one of
    /// GL_UNSIGNED_BYTE, GL_BYTE, ..
    ///
    /// If a non-zero named buffer object is bound to GL_PIXEL_UNPACK_BUFFER
    /// target while block of pixels is specified, data is treated as byte
    /// offset into buffer object's data store.
    //--------------------------------------------------------------------------
    static HandleGLError draw_pixels_to_frame_buffer(
      const Parameters& parameters);

    //--------------------------------------------------------------------------
    /// \brief Swaps the buffers of the current window if double buffered.
    /// \details This is on the user to swap buffers of the current window after
    /// running draw pixels, typically in the defined function passed into a
    /// GLUT display.
    /// \ref https://www.opengl.org/resources/libraries/glut/spec3/node21.html
    /// 4.6 glutSwapBuffers
    /// Performs buffer swap on layer in use for current window. Specifically,
    /// gluSwapBuffers promotes contents of the back buffer of layer in use of
    /// current window to become contents of front buffer. Contents of back
    /// buffer then become undefined. The update typically takes place during
    /// the vertical retrace of the monitor, rather than immediately after
    /// glutSwapBuffers is called.
    /// An implicit glFlush is done by glutSwapBuffers before it returns.
    /// If layer in use is not double buffered, glutSwapBuffers has no effect.
    //--------------------------------------------------------------------------
    inline static void swap_buffers()
    {
      glutSwapBuffers();
    }
};

} // namespace OpenGLInterface
} // namespace Visualization

#endif // VISUALIZATION_OPEN_GL_INTERFACE_DRAW_PIXELS_H
