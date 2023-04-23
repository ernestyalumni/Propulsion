#ifndef NUMERICAL_OPTIMIZATION_MINS_H
#define NUMERICAL_OPTIMIZATION_MINS_H

#include <math.h> // std::copysign
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
    static constexpr double GLIMIT {100.0};
    static constexpr double TINY {1.0e-20};

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

    double fu {0.0};

    while (fb > fc)
    {
      // Keep returning here until we bracket.
      const double r {(bx - ax) * (fb - fc)};
      const double q {(bx - cx) * (fb -fa)};
      // Compute u by parabolic extrapolation from a, b, c. TINY is used to
      // prevent any possible division by zero.
      double u {bx - ((bx - cx) * q - (bx - ax) * r) /
          (2.0 * std::copysign(
            std::max(std::abs(q-r), TINY),
            q - r))};

      const double ulimit {bx + GLIMIT * (cx - bx)};

      // We won't go farther than this. Test various possibilities:
      // Parabolic u is between b and c: try it.
      if ((bx - u) * (u - cx) > 0.0)
      {
        fu = func(u);
        // Got a minimum between b and c.
        if (fu < fc)
        {
          ax = bx;
          bx = u;
          fa = fb;
          fb = fu;
          return;
        }
        // Got a minimum between a and u
        else if (fu > fb)
        {
          cx = u;
          fc = fu;
          return;
        }
        // Parabolic fit was no use. Use default magnification.
        u = cx + GOLD * (cx - bx);
        fu = func(u);
      }
      // Parabolic fit is between its allowed limit.
      // cx < u < ulimit or ulimit < u < cx
      else if ((cx - u) * (u - ulimit) > 0.0)
      {
        fu = func(u);
        if (fu < fc)
        {
          shift3(bx, cx, u, u + GOLD * (u - cx));
          shift3(fb, fc, fu, func(u));
        }
      }
      // Limit parabolic u to maximum allowed value.
      else if ((u - ulimit) * (ulimit -cx) >= 0.0)
      {
        u = ulimit;
        fu = func(u);
      }
      // Reject parabolic u, use default magnification.
      else
      {
        u = cx + GOLD * (cx - bx);
        fu = func(u);
      }
      // Eliminate oldest point and continue.
      shift3(ax, bx, cx, u);
      shift3(fa, fb, fc, fu);
    }
  }

  inline static void shift2(double& a, double& b, const double c)
  {
    a = b;
    b = c;
  }

  inline static void shift3(double& a, double& b, double& c, const double d)
  {
    a = b;
    b = c;
    c = d;
  }

  inline static void move3(
    double& a,
    double& b,
    double& c,
    const double d,
    const double e,
    const double f)
  {
    a = d;
    b = e;
    c = f;
  }
};

} // namespace Optimization
} // namespace Numerical

#endif // NUMERICAL_OPTIMIZATION_MINS_H
