#ifndef VISUALIZATION_GLUT_INTERFACE_GLUT_WINDOW_H
#define VISUALIZATION_GLUT_INTERFACE_GLUT_WINDOW_H

#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <string_view>

namespace Visualization
{
namespace GLUTInterface
{

class GLUTWindow
{
  public:

    struct Parameters
    {
      //------------------------------------------------------------------------
      /// https://www.opengl.org/resources/libraries/glut/spec3/node12.html
      /// GLUT_RGBA - Bit mask to select an RGBA mode window. This is the
      /// default if neither GLUT_RGBA nor GLUT_INDEX are specified.
      /// GLUT_RGB - alias for GLUT_RGBA
      /// GLUT_INDEX - bit mask to select color index mode window. This
      /// overrides GLUT_RGBA if also specified.
      /// GLUT_SINGLE - bit mask to select a single buffered window. This
      /// overrides GLUT_SINGLE if it's also specified.
      /// GLUT_DOUBLE - bit mask to select a double buffered window. This
      /// overrides GLUT_SINGLE if it's also specified.
      //------------------------------------------------------------------------

      static std::array<unsigned int, 12> valid_modes_;

      //------------------------------------------------------------------------
      /// We considered also C-style variadic arguments, but from the examples
      /// it seemed like you needed to know the total number of variables
      /// beforehand:
      /// https://en.cppreference.com/w/cpp/utility/variadic
      /// From https://stackoverflow.com/questions/15465543/why-use-variadic-arguments-now-when-initializer-lists-are-available
      /// values in an initializer list are const objects from the standard.
      //------------------------------------------------------------------------
      Parameters(std::initializer_list<unsigned int> l);

      Parameters(
        const std::string_view display_name,
        const std::size_t width,
        const std::size_t height,
        std::initializer_list<unsigned int> l);

      void set_modes(std::initializer_list<unsigned int> l);

      char* get_display_name() const;

      unsigned int modes_;

      std::string_view display_name_;

      std::size_t width_;
      std::size_t height_;
    };

    GLUTWindow() = default;

    //--------------------------------------------------------------------------
    /// Non-copyable, non-moveable towards making this a singleton.
    //--------------------------------------------------------------------------
    GLUTWindow(const GLUTWindow&) = delete;
    GLUTWindow(GLUTWindow&&) = delete;
    GLUTWindow& operator=(const GLUTWindow&) = delete;
    GLUTWindow& operator=(GLUTWindow&& other) = delete;

    virtual ~GLUTWindow() = default;

    //--------------------------------------------------------------------------
    /// \url https://www.opengl.org/resources/libraries/glut/spec3/node10.html
    /// \details Runs
    /// glutInit
    /// void glutInit(int *argcp, char **argv);
    /// X Window System specific options parsed by glutInit as follows:
    /// -display DISPLAY - Specify X server to connect to. If not specified,
    /// value of DISPLAY environment variable used.
    /// -geoemetry W x H + X + Y - Determines where windows's should be created
    /// on screen. Parameter following -geometry should be formatted as a
    /// standard X geometry specification. The effect of using this option is to
    /// change the GLUT initial size and initial position the same as if
    /// glutInitWindowSize or glutInitWindowPosition were called directly.
    ///
    /// https://www.opengl.org/resources/libraries/glut/spec3/node12.html
    /// glutInitDisplayMode sets initial display mode.
    //--------------------------------------------------------------------------
    static void initialize_glut(
      int* argcp,
      char** argv,
      const Parameters& parameters);

    inline static int get_window_identifier()
    {
      return window_identifier_;
    }

    //--------------------------------------------------------------------------
    /// \brief Singleton instance of GLUTWindow.
    //--------------------------------------------------------------------------
    static GLUTWindow& instance();

  private:

    //--------------------------------------------------------------------------
    /// https://www.opengl.org/resources/libraries/glut/spec3/node16.html
    /// Unique small integer identifier for the window. Range of allocated
    /// identifiers starts at one. This window identifier can be used when
    /// calling glutSetWindow.
    ///
    /// \details Make this static for the purpose towards making this a
    /// singleton.
    //--------------------------------------------------------------------------
    static int window_identifier_;

    static std::unique_ptr<GLUTWindow> instance_;

    //--------------------------------------------------------------------------
    /// \ref https://en.cppreference.com/w/cpp/thread/mutex
    /// \ref https://refactoring.guru/design-patterns/singleton/cpp/example#example-1
    /// A calling thread owns a mutex from time it successfully calls either
    /// lock or try_lock until it calls unlock.
    /// When thread owns a mutex, all other threads will block (for calls to
    /// lock) or receive false return value (for try_lock) if they attempt to
    /// claim ownership of mutex.
    //--------------------------------------------------------------------------
    static std::mutex mutex_;
};

} // namespace GLUTInterface
} // namespace Visualization

#endif // VISUALIZATION_GLUT_INTERFACE_GLUT_WINDOW_H
