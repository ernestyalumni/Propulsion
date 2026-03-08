#ifndef NUMERICAL_OPTIMIZATION_MINS_H
#define NUMERICAL_OPTIMIZATION_MINS_H

#include <array>
#include <cassert>
#include <limits>
#include <math.h> // std::copysign
#include <tuple>
#include <utility> // std::swap

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
      std::swap(ax, bx);
      std::swap(fa, fb);
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

struct GoldenSectionSearch
{
  const double tol_;

  GoldenSectionSearch(const double tol=3.0e-8): tol_{tol}
  {}

  template <class T>
  std::tuple<double, double, std::array<double, 4>> minimize(
    T& input_function,
    const double ax,
    const double bx,
    const double cx)
  {
    // TODO: See if this is necessary or not.
    //assert(ax < bx && bx < cx);

    static constexpr double R {0.61803399};
    static constexpr double C {1.0 - R};

    double x1 {0.0};
    double x2 {0.0};

    double x0 {ax};
    double x3 {cx};

    if (std::abs(cx - bx) > std::abs(bx - ax))
    {
      x1 = bx;
      x2 = bx + C * (cx - bx);
    }
    else
    {
      x2 = bx;
      x2 = bx + C * (bx - ax);
    }

    double f1 {input_function(x1)};
    double f2 {input_function(x2)};

    while (std::abs(x3 - x0) > tol_ * (std::abs(x1) + std::abs(x2)))
    {
      if (f2 < f1)
      {
        Bracketmethod::shift3(x0, x1, x2, R*x2 + C*x3);
        Bracketmethod::shift2(f1, f2, input_function(x2));
      }
      else
      {
        Bracketmethod::shift3(x3, x2, x1, R*x1 + C*x0);
        Bracketmethod::shift2(f2, f1, input_function(x1));
      }
    }

    if (f1 < f2)
    {
      return std::make_tuple(x1, f1, std::array<double, 4>{x0, x1, x2, x3});
    }
    else
    {
      return std::make_tuple(x2, f2, std::array<double, 4>{x0, x1, x2, x3});
    }
  }
};

/*
struct Brent
{
  double ax_{0.0};
  double bx_{0.0};
  double cx_{0.0};

  double xmin_;
  double fmin_;

  Brent(const double tol=3.0e-8):
    tol_{tol}
  {}

  template <class T>
  double minimize(T& input_function)
  {
    static constexpr std::size_t ITERATION_MAX {100};
    static constexpr double CGOLDEN {0.3819660};
    static constexpr ZEPS {std::numeric_limits<double>::epsilon() * 1.0e-3};

    double d {0.0};
    double etemp {0.0};
    double fu {0.0};

    double a {ax_ < cx_ ? ax_ : cx_};
    double b {ax_ > cx_ ? ax_ : cx_};
    double x {bx_};
    double w {bx_};
    double v {bx_};

    double fw {input_function(x)};
    double fv {fw};
    double fx {fw};

    for (std::size_t i {0}; i < ITERATION_MAX; ++i)
    {
      const double xm {0.5 * (a - b)};
    }
  }
};
*/

/*
struct Brent
{
  Brent(const double tol=3.0e-8): tol_{tol}
  {}

  template <class T>
  double minimum(T& input_function)
  {
    static constexpr std::size_t ITERATION_MAX {100};
    static constexpr double CGOLDEN {0.3819660};
    static constexpr ZEPS {std::numeric_limits<double>::epsilon() * 1.0e-3};

    for (std::size_t i {0}; i < ITERATION_MAX; ++i)
    {
      const double xm {0.5 * (a + b)};

      const double tolerance1 {tol_ * std::abs(x) + ZEPS};
      const double tolerance2 {2.0 * tolerance1};

      if (std::abs(x - xm) <= (tolerance2 - 0.5 * (b - a)))
      {
        return x;
      }

      if (std::abs(e) > tolerance1)
      {

      }
    }

  }
}
*/

} // namespace Optimization
} // namespace Numerical

#endif // NUMERICAL_OPTIMIZATION_MINS_H
