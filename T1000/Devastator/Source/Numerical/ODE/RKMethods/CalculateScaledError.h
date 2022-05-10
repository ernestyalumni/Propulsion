#ifndef NUMERICAL_ODE_RK_METHODS_CALCULATE_SCALED_ERROR_H
#define NUMERICAL_ODE_RK_METHODS_CALCULATE_SCALED_ERROR_H

#include "Algebra/Modules/Vectors/NVector.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <valarray>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

template <typename Field = double>
class CalculateScaledError
{
  public:

    CalculateScaledError() = delete;
    CalculateScaledError(const Field atol, const Field rtol):
      a_tolerance_{atol},
      r_tolerance_{rtol}
    {}

    //--------------------------------------------------------------------------
    /// \param y_err Error calculated from the delta coefficients with k
    /// coefficients.
    //--------------------------------------------------------------------------
    template <typename ContainerT, std::size_t N>
    Field operator()(
      const ContainerT& y_0,
      const ContainerT& y_out,
      const ContainerT& y_err)
    {
      Field error {static_cast<Field>(0)};

      for (std::size_t i {0}; i < N; ++i)
      {
        const Field scale {
          a_tolerance_ +
            r_tolerance_ * std::max(std::abs(y_0[i]), std::abs(y_out[i]))};

        error += (y_err[i] / scale) * (y_err[i] / scale);
      }

      return std::sqrt(error / N);
    }
}

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_CALCULATE_SCALED_ERROR_H
