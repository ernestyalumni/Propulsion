#ifndef NUMERICAL_ODE_RK_METHODS_CALCULATE_ERROR_H
#define NUMERICAL_ODE_RK_METHODS_CALCULATE_ERROR_H

#include "Coefficients/BCoefficients.h"
#include "Coefficients/KCoefficients.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

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

    template <std::size_t N>
    Field calculate_scaled_error(
      const ContainerT& y_0,
      const ContainerT& y_out,
      const Coefficients::KCoefficients<S, ContainerT>& k_coefficients,
      const Field h)
    {
      Field error {static_cast<Field>(0)};
      Field error2 {static_cast<Field>(0)};
      Field error_i {static_cast<Field>(0)};

      for (std::size_t i {0}; i < N; ++i)
      {
        const Field scale {
          a_tolerance_[i] +
            r_tolerance_[i] * std::max(std::abs(y_0[i]), std::abs(y_out[i]))};

        error_i = (y_out[i] - y_0[i]) / h -
          bhh_coefficients_.get_ith_element(1) *
            k_coefficients.get_ith_coefficient(1)[i] -
          bhh_coefficients_.get_ith_element(2) *
            k_coefficients.get_ith_coefficient(9)[i] -
          bhh_coefficients_.get_ith_element(3) *
            k_coefficients.get_ith_coefficient(12)[i];

        error2 += (error_i / scale) * (error_i / scale);

        error_i = static_cast<Field>(0);

        for (std::size_t j {2}; j <= S; ++j)
        {
          error_i += delta_coefficients_.get_ith_element(j) *
            k_coefficients.get_ith_coefficient(j)[i];
        }

        error += (error_i / scale) * (error_i / scale);
      }

      const Field denominator {error + 0.01 * error2};

      return std::abs(h) * error * std::sqrt(1.0 / (N * denominator));
    }

    template <std::size_t N>
    Field calculate_scaled_error(
      const ContainerT& y_0,
      const ContainerT& y_out,
      const ContainerT& y_err,
      const ContainerT& y_err2,
      const Field h)
    {
      Field error {static_cast<Field>(0)};
      Field error2 {static_cast<Field>(0)};

      for (std::size_t i {0}; i < N; ++i)
      {
        const Field scale {
          a_tolerance_[i] +
            r_tolerance_[i] * std::max(std::abs(y_0[i]), std::abs(y_out[i]))};

        error += (y_err[i] / scale) * (y_err[i] / scale);
        error2 += (y_err2[i] / scale) * (y_err2[i] / scale);
      }

      const Field denominator {error2 + 0.01 * error};

      return std::abs(h) * error2 * std::sqrt(1.0 / (N * denominator));
    }

  protected:

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
      y_err = (y_out - y_in) / h - bhh_coefficients_.get_ith_element(1) *
        k_coefficients.get_ith_coefficient(1) -
        bhh_coefficients_.get_ith_element(2) *
          k_coefficients.get_ith_coefficient(9) -
        bhh_coefficients_.get_ith_element(3) *
          k_coefficients.get_ith_coefficient(12);

      y_err2 = delta_coefficients_.get_ith_element(1) *
        k_coefficients.get_ith_coefficient(1);

      for (std::size_t j {2}; j <= S; ++j)
      {
        y_err2 += delta_coefficients_.get_ith_element(j) *
          k_coefficients.get_ith_coefficient(j);
      }
    }

  private:

    const Coefficients::DeltaCoefficients<S, Field>& delta_coefficients_;
    const Coefficients::BCoefficients<BHHSize, Field>& bhh_coefficients_;
    ContainerT a_tolerance_;
    ContainerT r_tolerance_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_CALCULATE_ERROR_H
