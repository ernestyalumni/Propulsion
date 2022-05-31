#ifndef NUMERICAL_ODE_RK_METHODS_CALCULATE_ERROR_H
#define NUMERICAL_ODE_RK_METHODS_CALCULATE_ERROR_H

#include "Algebra/Modules/Vectors/NVector.h"
#include "Coefficients/BCoefficients.h"
#include "Coefficients/KCoefficients.h"

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

template <
  std::size_t S,
  std::size_t BHHSize,
  typename ContainerT,
  typename Field = double>
class CalculateError
{
  public:

    CalculateError() = delete;
    CalculateError(
      const Coefficients::DeltaCoefficients<S, Field>& delta_coefficients,
      const Coefficients::BCoefficients<BHHSize, Field>& bhh_coefficients,
      const ContainerT& atol,
      const ContainerT& rtol
      ):
      delta_coefficients_{delta_coefficients},
      bhh_coefficients_{bhh_coefficients},
      a_tolerance_{atol},
      r_tolerance_{rtol}
    {}

    //--------------------------------------------------------------------------
    /// \ref https://github.com/blackstonep/Numerical-Recipes/blob/master/stepperdopr853.h
    /// \details Calculates yerr, yerr2, or what would be err1 in Hairer and
    /// Wanner's code.
    //--------------------------------------------------------------------------
    void calculate_weighted_differences(
      const ContainerT& y_in,
      const ContainerT& y_out,
      const Field h,
      const Coefficients::KCoefficients<S, ContainerT>& k_coefficients,
      ContainerT& y_err,
      ContainerT& y_err2)
    {
      yerr = (y_out - y_in) / h - bhh_coefficients_.get_ith_element(1) *       
    }

    //--------------------------------------------------------------------------
    /// \param y_err Error calculated from the delta coefficients with k
    /// coefficients.
    //--------------------------------------------------------------------------
    template <typename ContainerT, std::size_t N>
    Field operator()(
      const ContainerT& y_0,
      const ContainerT& y_out,
      const ContainerT& y_err) const
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

  private:

    Coefficients::DeltaCoefficients<S, Field>& delta_coefficients_;
    Coefficients::BCoefficients<BHHSize, Field>& bhh_coefficients_;
    ContainerT a_tolerance_;
    ContainerT r_tolerance_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_CALCULATE_ERROR_H
