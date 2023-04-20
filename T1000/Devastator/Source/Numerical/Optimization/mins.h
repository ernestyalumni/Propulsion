#ifndef NUMERICAL_OPTIMIZATION_MINS_H
#define NUMERICAL_OPTIMIZATION_MINS_H

#include <utility> // std::swap

using std::swap;

namespace Numerical
{
namespace Optimization
{

//------------------------------------------------------------------------------
/// \ref Numerical Recipes, 3rd. Ed. Press, Teukolsky, Vetterling, Flannery.
/// 10.1 Initially Bracketing a Minimum. pp. 491
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// \brief Base class for 1-dim. minimization routines. Provides routine to
/// bracket a minimum and several utility functions.
//------------------------------------------------------------------------------
struct Bracketmethod
{
  double ax;
  double bx;
  double cx;
  double fa;
  double fb;
  double fc;

  template <class T>
  void bracket(const double a, const double b, T& func)
  {
    static constexpr double GOLD {1.618034};

    ax = a;
    bx = b;

    fa = func(ax);
    fb = func(fb);  

    if (fb > fa)
    {
      swap(ax, bx);
      swap(fa, fb);
    }

    // First guess for c.
    cx = bx + GOLD * (bx - ax);

    fc = func(cx);

    while (fb > fc)
    {
      // Keep returning here until we bracket.
      const double r {(bx - ax) * (fb - fc)};
      const double q {(bx - cx) * (fb -fa)};
      // Compute u by parabolic extrapolation from a, b, c. TINY is used to
      // prevent any possible division by zero.
      const double u {bx - ((bx - cx) * q - (bx - ax) * r) / (2.0 * SIGN * (
        std::max(std::abs(q-r), TINY), q - r))};
    }
  }
};

} // namespace Optimization
} // namespace Numerical

#endif // NUMERICAL_OPTIMIZATION_MINS_H
