#ifndef NUMERICAL_ODE_RK_METHODS_CALCULATE_NEW_Y_H
#define NUMERICAL_ODE_RK_METHODS_CALCULATE_NEW_Y_H

#include "Algebra/Modules/Vectors/NVector.h"
#include "Coefficients/ACoefficients.h"
#include "Coefficients/BCoefficients.h"
#include "Coefficients/CCoefficients.h"
#include "Coefficients/KCoefficients.h"

#include <algorithm> // std::copy, std::transform
#include <array>
#include <cstddef>
#include <initializer_list>
#include <utility> // std::forward
#include <valarray>

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
class CalculateNewY
{
  public:

    template <std::size_t M>
    using NVector = Algebra::Modules::Vectors::NVector<M, Field>;

    CalculateNewY() = delete;

    CalculateNewY(
      DerivativeType&& derivative,
      const Coefficients::ACoefficients<S, Field>& a_coefficients,
      const Coefficients::BCoefficients<S, Field>& b_coefficients,
      const Coefficients::CCoefficients<S, Field>& c_coefficients
      ):
      // Forward rvalues as rvalues and prohibits forwarding of rvalues as
      // lvalues.
      derivative_{std::forward<DerivativeType>(derivative)},
      a_coefficients_{a_coefficients},
      b_coefficients_{b_coefficients},
      c_coefficients_{c_coefficients}
    {}

    CalculateNewY(
      DerivativeType& derivative,
      const Coefficients::ACoefficients<S, Field>& a_coefficients,
      const Coefficients::BCoefficients<S, Field>& b_coefficients,
      const Coefficients::CCoefficients<S, Field>& c_coefficients
      ):
      // Forward lvalues as either lvalues or as rvalues.
      derivative_{std::forward<DerivativeType>(derivative)},
      a_coefficients_{a_coefficients},
      b_coefficients_{b_coefficients},
      c_coefficients_{c_coefficients}
    {}

    virtual ~CalculateNewY() = default;

    std::valarray<Field> calculate_new_y(
      const Field h,
      const Field x,
      const std::valarray<Field>& y,
      const std::valarray<Field>& initial_dydx,
      Coefficients::KCoefficients<S, std::valarray<Field>>& k_coefficients)
    {
      std::valarray<Field> y_out;

      k_coefficients.ith_coefficient(1) = initial_dydx;

      for (std::size_t l {2}; l <= S; ++l)
      {
        const Field x_l {x + get_c_i(l) * h};

        // y_out = y + h * (a_l1 * k_1 + ... + a_l,l-1 * k_{l-1})
        y_out = y + sum_a_and_k_products(k_coefficients, l, h);

        k_coefficients.ith_coefficient(l) = derivative_(x_l, y_out);
      }

      y_out = get_b_i(1) * k_coefficients.ith_coefficient(1);

      for (std::size_t j {2}; j <= S; ++j)
      {
        y_out += get_b_i(j) * k_coefficients.ith_coefficient(j);
      }

      y_out *= h;

      return y_out + y;
    }

    template <std::size_t N>
    NVector<N> calculate_new_y(
      const Field h,
      const Field x,
      const NVector<N>& y,
      const NVector<N>& initial_dydx,
      Coefficients::KCoefficients<S, NVector<N>>& k_coefficients)
    {
      NVector<N> y_out {};
      k_coefficients.ith_coefficient(1) = initial_dydx;

      for (std::size_t l {2}; l <= S; ++l)
      {
        const Field x_l {x + get_c_i(l) * h};

        // y_out = y + h * (a_l1 * k_1 + ... + a_l,l-1 * k_{l-1})
        y_out = y + sum_a_and_k_products<N>(k_coefficients, l, h);

        k_coefficients.ith_coefficient(l) = derivative_(x_l, y_out);
      }

      y_out = get_b_i(1) * k_coefficients.ith_coefficient(1);

      for (std::size_t j {2}; j <= S; ++j)
      {
        y_out += get_b_i(j) * k_coefficients.ith_coefficient(j);
      }

      y_out *= h;

      return y_out + y;
    }

    template <typename ContainerT>
    ContainerT calculate_derivative(
      const Field x,
      const ContainerT& y)
    {
      return derivative_(x, y);
    }

  protected:

    std::valarray<Field> sum_a_and_k_products(
      Coefficients::KCoefficients<S, std::valarray<Field>>& k_coefficients,
      const std::size_t l,
      const Field h)
    {
      std::valarray<Field> a_lj_times_kj {
        get_a_ij(l, 1) * k_coefficients.ith_coefficient(1)};

      for (std::size_t j {2}; j < l; ++j)
      {
        a_lj_times_kj += get_a_ij(l, j) * k_coefficients.ith_coefficient(j);
      }

      return h * a_lj_times_kj;
    }

    template <std::size_t N>
    NVector<N> sum_a_and_k_products(
      Coefficients::KCoefficients<S, NVector<N>>& k_coefficients,
      const std::size_t l,
      const Field h)
    {
      NVector<N> a_lj_times_kj {
        get_a_ij(l, 1) * k_coefficients.ith_coefficient(1)};

      for (std::size_t j {2}; j < l; ++j)
      {
        a_lj_times_kj += get_a_ij(l, j) * k_coefficients.ith_coefficient(j);
      }

      return h * a_lj_times_kj;
    }

    Field get_a_ij(const std::size_t i, const std::size_t j) const
    {
      return a_coefficients_.get_ij_element(i, j);
    }

    Field get_b_i(const std::size_t i) const
    {
      return b_coefficients_.get_ith_element(i);
    }

    Field get_c_i(const std::size_t i) const
    {
      return c_coefficients_.get_ith_element(i);
    }

  private:

    DerivativeType derivative_;

    const Coefficients::ACoefficients<S, Field>& a_coefficients_;
    const Coefficients::BCoefficients<S, Field>& b_coefficients_;
    const Coefficients::CCoefficients<S, Field>& c_coefficients_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_CALCULATE_NEW_Y_H
