#ifndef NUMERICAL_INTERPOLATION_LINEAR_INTERPOLATION_1D_H
#define NUMERICAL_INTERPOLATION_LINEAR_INTERPOLATION_1D_H

#include "Interpolation1d.h"

namespace Numerical
{
namespace Interpolation
{

//------------------------------------------------------------------------------
/// \brief Piecewise linear interpolation object. Construct with x and y
///   vectors, then call interp for interpolated values.
//------------------------------------------------------------------------------
template <classname TContainer>
struct Linear_interp : Base_interp
{
  Linear_interp(TContainer& xv, TContainer& yv) :
    Base_interp(xv, &yv[0], 2)
  {}

  double rawinterp(int j, double x)
  {
    // Table is defective, but we can recover.
    if (x_[j] == x_[j + 1])
    {
      return y_[j];
    }
    else
    {
      return y_[j] + ((x - x_[j]) / (x_[j + 1] - x_[j])) * (y_[j + 1] - y_[j]);
    }
  }
}

} // namespace Interpolation
} // namespace Numerical

#endif // NUMERICAL_INTERPOLATION_LINEAR_INTERPOLATION_1D_H