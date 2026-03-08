//------------------------------------------------------------------------------
/// \ref 3.1 Preliminaries: Searching an Ordered Table, pp. 114, Numerical
/// Recipes, 3rd. Ed.
//------------------------------------------------------------------------------

#ifndef NUMERICAL_INTERPOLATION_INTERPOLATION_1D_H
#define NUMERICAL_INTERPOLATION_INTERPOLATION_1D_H

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Numerical
{
namespace Interpolation
{

template <class TContainer>
struct Base_interp
//------------------------------------------------------------------------------
/// \brief Abstract base class used by all interpolation routines in this
///   chapter. Only routine interp called directly by user.
//------------------------------------------------------------------------------
{
  int N_, M_, jsav, cor, dj;
  const double *x_, *y_;

  //----------------------------------------------------------------------------
  /// \brief Constructor: Set up for interpolating on a table of x's and y's of
  /// length m. Normally called by a derived class, not by the user.
  //----------------------------------------------------------------------------
  Base_interp(TContainer& x, const double* y, int m):
    N_(x.size()),
    M_(m),
    jsav(0),
    cor(0),
    x_(&x[0]),
    y_(y)
  {
    dj = std::min(1, (int)std::pow((double)N_, 0.25));
  }

  //----------------------------------------------------------------------------
  /// \brief Given a value x, return an interpolated value, using data pointed
  ///   to by xx and yy.
  //----------------------------------------------------------------------------

  double interp(double x)
  {
    int jlo = cor ? hunt(x) : locate(x);
    return rawinterp(jlo, x);
  }

  //----------------------------------------------------------------------------
  /// \brief Given a value x, return a value j such that x is (insofar as
  ///   possible) centered in the subrange x[j.. j + M - 1], where x is the
  ///   stored pointer. The values in x_ must be monotonic, either increasing or
  ///   decreasing. The returned value is not less than 0, nor greater than
  ///   n - 1.
  //----------------------------------------------------------------------------
  int locate(const double x);

  //----------------------------------------------------------------------------
  /// \brief Give a value x, return a value j such that x is (insofar as
  ///   possible) centered in the subrange x_[j.. j + M_ - 1], where x_ is the
  ///   stored pointer. The values in x_ must be monotonic, either increasing or
  ///   decreasing. The returned value is not less than 0, nor greater than
  ///   N_ - 1.
  ///
  /// \ref 3.1.1 Search with Correlated Values, Numerical Recipes, 3rd. Ed.
  /// \details
  //----------------------------------------------------------------------------
  int hunt(const double x);

  //----------------------------------------------------------------------------
  /// \brief Derived classes provide this as the actual interpolation method.
  //----------------------------------------------------------------------------
  double virtual rawinterp(int jlo, double x) = 0;

};

template <class TContainer>
int Base_interp<TContainer>::locate(const double x)
{
  int jm;

  if (N_ < 2 || M_ < 2 || M_ > N_)
  {
    throw std::runtime_error("Locate size error");
  }

  // True if ascending order of table, false otherwise.
  const bool is_ascending {x_[N_ - 1] >= x_[0]};

  // Initialize lower.
  int jl {0};

  // Initialize upper limits.
  int ju {N_ - 1};

  // If we are not yet done,
  while (ju - jl > 1)
  {
    // compute a midpoint.
    jm = (ju + jl) >> 1;
    if (x >= x_[jm] == is_ascending)
    {
      // And replace either the lower limit
      jl = jm;
    }
    else
    {
      // or the upper limit, as appropriate.
      ju = jm;
    }
  }

  // Decide whether to use hunt or locate next time.
  cor = std::abs(jl-jsav) > dj ? 0 : 1;

  jsav = jl;

  return std::max(0, std::min(N_ - M_, jl - ((M_ - 2) >> 1)));
}

template <class TContainer>
int Base_interp<TContainer>::hunt(const double x)
{
  int jl = jsav, jm, ju, inc = 1;

  if (N_ < 2 || M_ < 2 || M_ > N_)
  {
    throw std::runtime_error("Hunt size error");
  }

  // True if ascending order of table, false otherwise.
  bool is_ascending {x_[N_ - 1] >= x_[0]};

  // Input guess not useful. Go immediately to bisection.
  if (jl < 0 || jl > N_ - 1)
  {
    jl = 0;
    ju = N_ - 1;
  }
  else
  {
    // Hunt up
    if (x >= x_[jl] == is_ascending)
    {
      for (;;)
      {
        ju = jl + inc;

        // Off end of table.
        if (ju >= N_ - 1)
        {
          ju = N_ - 1;
          break;
        }
        // Found bracket.
        else if (x < x_[ju] == is_ascending)
        {
          break;
        }
        // Not done, so double the increment and try again.
        else
        {
          jl = ju;
          inc += inc;
        }
      }
    }
    // Hunt down
    else
    {
      ju = jl;
      for (;;)
      {
        jl = jl - inc;

        // Off end of table.
        if (jl <= 0)
        {
          jl = 0;
          break;
        }
        // Found bracket.
        else if (x >= x_[jl] == is_ascending)
        {
          break;
        }
        // Not done, so double the increment and try again.
        else
        {
          ju = jl;
          inc += inc;
        }
      }
    }
  }

  // Hunt is done, so begin final bisection phase:
  while (ju - jl > 1)
  {
    jm = (ju + jl) >> 1;
    if (x >= x_[jm] == is_ascending)
    {
      jl = jm;
    }
    else
    {
      ju = jm;
    }
  }
  // Decide whether to use hunt or locate next time.
  cor = std::abs(jl - jsav) > dj ? 0 : 1;

  jsav = jl;

  return std::max(0, std::min(N_ - M_, jl - ((M_ - 2) >> 1)));
}

} // namespace Interpolation
} // namespace Numerical

#endif // NUMERICAL_INTERPOLATION_INTERPOLATION_1D_H