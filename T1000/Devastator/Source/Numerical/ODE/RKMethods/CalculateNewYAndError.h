#ifndef NUMERICAL_ODE_RK_METHODS_CALCULATE_NEW_Y_AND_ERROR_H
#define NUMERICAL_ODE_RK_METHODS_CALCULATE_NEW_Y_AND_ERROR_H

#include "Coefficients/ACoefficients.h"
#include "Coefficients/BCoefficients.h"
#include "Coefficients/CCoefficients.h"

#include <algorithm> // std::transform
#include <array>
#include <cstddef>
#include <initializer_list>
#include <utility> // std::forward

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

//------------------------------------------------------------------------------
/// \tparam S - number of stages.
//------------------------------------------------------------------------------
template <std::size_t S, typename DerivativeType, typename Field = double>
class CalculateNewYAndError
{
  public:

    CalculateNewYAndError() = delete;
    CalculateNewYAndError(
      DerivativeType&& derivative,
      const Coefficients::ACoefficients<S, Field>& a_coefficients,
      const Coefficients::CCoefficients<S, Field>& c_coefficients,
      const Coefficients::DeltaCoefficients<S, Field>& delta_coefficients
      ):
      // Forward rvalues as rvalues and prohibits forwarding of rvalues as
      // lvalues.
      derivative_{std::forward<DerivativeType>(derivative)},
      a_coefficients_{a_coefficients},
      c_coefficients_{c_coefficients},
      delta_coefficients_{delta_coefficients}
    {}

    CalculateNewYAndError(
      DerivativeType& derivative,
      const Coefficients::ACoefficients<S, Field>& a_coefficients,
      const Coefficients::CCoefficients<S, Field>& c_coefficients,
      const Coefficients::DeltaCoefficients<S, Field>& delta_coefficients
      ):
      // Forward lvalues as either lvalues or as rvalues.
      derivative_{std::forward<DerivativeType>(derivative)},
      a_coefficients_{a_coefficients},
      c_coefficients_{c_coefficients},
      delta_coefficients_{delta_coefficients}
    {}

    template <typename ContainerT>
    void operator()(
      const Field h,
      const Field x,
      const ContainerT& y,
      const ContainerT& initial_dydx,
      Coefficients::BCoefficients<S, ContainerT>& k_coefficients,
      ContainerT& y_out)
    {
      k_coefficients[0] = initial_dydx;

      for (std::size_t l {2}; l <= S; ++l)
      {
        const Field x_l {x + get_c_i(l) * h};

        sum_a_and_k_products<ContainerT>(k_coefficients, l, y_out);

        // y_out = y + h * (a_l1 * k_1 + ... + a_l,l-1 * k_{l-1})
        std::transform(
          y_out.begin(),
          y_out.end(),
          y.begin(),
          y_out.begin(),
          std::plus<Field>());

        // k_l = f(x + c_l * h, y + h * (a_l1 * k_1 + ... + a_l,l-1 * k_{l-1}))
        derivative_(x_l, y_out, k_coefficients[l - 1]);        
      }
    }

    template <typename ContainerT>
    void calculate_error(
      const Field h,
      const Coefficients::BCoefficients<S, ContainerT>& k_coefficients,
      ContainerT& y_err)
    {
      ContainerT delta_l_times_k_l {};

      // delta_1 * k_1
      std::transform(
        k_coefficients.get_ith_element(1).begin(),
        k_coefficients.get_ith_element(1).end()),
        delta_l_times_k_l.begin(),
        std::bind(
          std::multiplies<Field>(),
          std::placeholders::_1,
          delta_coefficients_.get_ith_element(1));

      y_err = delta_l_times_k_l;

      for (std::size_t j {2}; j <= S; ++j)
      {
        // Calculate delta_j * k_j
        std::transform(
          k_coefficients.get_ith_element(j).begin(),
          k_coefficients.get_ith_element(j).end(),
          delta_l_times_k_l.begin(),
          std::bind(
            std::multiplies<Field>(),
            std::placeholders::_1,
            delta_coefficients_.get_ith_element(j)));

        // Calculate y_err += y_err + delta_j * k_j
        std::transform(
          y_err.begin(),
          y_err.end(),
          delta_l_times_k_l.begin(),
          y_err.begin(),
          std::plus<Field>());
      }

      std::transform(
        y_err.begin(),
        y_err.end(),
        y_err.begin(),
        std::bind(
          std::multiplies<Field>(),
          std::placeholders::_1,
          h));
    }

  protected:

    template <typename ContainerT>
    void sum_a_and_k_products(
      Coefficients::BCoefficients<S, ContainerT>& k_coefficients,
      const std::size_t l,
      const Field h,
      ContainerT& out)
    {
      // j = 1...l-1
      ContainerT a_lj_times_k_j;

      std::transform(
        k_coefficients.get_ith_element(1).begin(),
        k_coefficients.get_ith_element(1).end(),
        a_lj_times_k_j.begin(),
        std::bind(
          std::multiplies<Field>(),
          std::placeholders::_1,
          get_a_ij(l, 1)));

      out = a_lj_times_k_j;

      for (std::size_t j {2}; j < l; ++j)
      {
        // Calculate a_lj * k_j
        std::transform(
          k_coefficients.get_ith_element(j).begin(),
          k_coefficients.get_ith_element(j).end(),
          a_lj_times_k_j.begin(),
          std::bind(
            std::multiplies<Field>(),
            std::placeholders::_1,
            get_a_ij(l, j)));

        // out = out + a_lj_times_k_j
        std::transform(
          out.begin(),
          out.end(),
          a_lj_times_k_j.begin(),
          out.begin(),
          std::plus<Field>());
      }

      // Calculate h * (a_l1 * k_1 + ... + a_l,l-1 * k_l-1)
      std::transform(
        out.begin(),
        out.end(),
        out.begin(),
        std::bind(
          std::multiplies<Field>(),
          std::placeholders::_1,
          h));
    }

    Field get_a_ij(const std::size_t i, const std::size_t j) const
    {
      return a_coefficients_.get_ij_element(i, j);
    }

    Field get_c_i(const std::size_t i) const
    {
      return c_coefficients_.get_ith_element(i);
    }

  private:

    DerivativeType derivative_;

    const Coefficients::ACoefficients<S, Field>& a_coefficients_;
    const Coefficients::CCoefficients<S, Field>& c_coefficients_;
    const Coefficients::DeltaCoefficients<S, Field>& delta_coefficients_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_CALCULATE_NEW_Y_AND_ERROR_H
