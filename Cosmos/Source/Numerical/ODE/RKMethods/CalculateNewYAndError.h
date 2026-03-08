#ifndef NUMERICAL_ODE_RK_METHODS_CALCULATE_NEW_Y_AND_ERROR_H
#define NUMERICAL_ODE_RK_METHODS_CALCULATE_NEW_Y_AND_ERROR_H

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
class CalculateNewYAndError
{
  public:

    template <std::size_t M>
    using NVector = Algebra::Modules::Vectors::NVector<M, Field>;

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

    virtual ~CalculateNewYAndError() = default;

    //--------------------------------------------------------------------------
    /// \tparam N Dimension of out or of Container T, or of y in the ODE.
    //--------------------------------------------------------------------------
    template <typename ContainerT, std::size_t N>
    void calculate_new_y(
      const Field h,
      const Field x,
      const ContainerT& y,
      const ContainerT& initial_dydx,
      Coefficients::KCoefficients<S, ContainerT>& k_coefficients,
      ContainerT& y_out)
    {
      k_coefficients.ith_coefficient(1) = initial_dydx;

      for (std::size_t l {2}; l <= S; ++l)
      {
        const Field x_l {x + get_c_i(l) * h};

        sum_a_and_k_products<ContainerT, N>(k_coefficients, l, h, y_out);

        // y_out = y + h * (a_l1 * k_1 + ... + a_l,l-1 * k_{l-1})
        std::transform(
          y_out.begin(),
          y_out.end(),
          y.begin(),
          y_out.begin(),
          std::plus<Field>());

        // k_l = f(x + c_l * h, y + h * (a_l1 * k_1 + ... + a_l,l-1 * k_{l-1}))
        derivative_(x_l, y_out, k_coefficients.ith_coefficient(l));
      }
    }

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

      return y_out;
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

      return y_out;
    }

    template <typename ContainerT>
    ContainerT calculate_derivative(
      const Field x,
      const ContainerT& y)
    {
      return derivative_(x, y);
    }

    template <typename ContainerT, std::size_t N>
    void calculate_new_y_and_error(
      const Field h,
      const Field x,
      const ContainerT& y,
      const ContainerT& initial_dydx,
      Coefficients::KCoefficients<S, ContainerT>& k_coefficients,
      ContainerT& y_out,
      ContainerT& y_err)
    {
      calculate_new_y<ContainerT, N>(
        h,
        x,
        y,
        initial_dydx,
        k_coefficients,
        y_out);
      calculate_error<ContainerT, N>(h, k_coefficients, y_err);
    }

    //--------------------------------------------------------------------------
    /// \details The error calculated in these cases is the difference between
    /// y_{n+1} and y^*_{n+1} \equiv \widehat{y}_{n+1}, i.e. y_{n+1} - y^*_{n+1}
    /// which is obtained directly from the "delta" coefficients.
    //--------------------------------------------------------------------------

    template <typename ContainerT, std::size_t N>
    void calculate_error(
      const Field h,
      const Coefficients::KCoefficients<S, ContainerT>& k_coefficients,
      ContainerT& y_err)
    {
      std::array<Field, N> delta_l_times_k_j {};

      k_coefficients.scalar_multiply(
        delta_l_times_k_j,
        1,
        delta_coefficients_.get_ith_element(1));

      std::copy(
        delta_l_times_k_j.begin(),
        delta_l_times_k_j.end(),
        y_err.begin());

      for (std::size_t j {2}; j <= S; ++j)
      {
        // Calculate delta_j * k_j
        k_coefficients.scalar_multiply(
          delta_l_times_k_j,
          j,
          delta_coefficients_.get_ith_element(j));

        // Calculate y_err += y_err + delta_j * k_j
        std::transform(
          y_err.begin(),
          y_err.end(),
          delta_l_times_k_j.begin(),
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

    template <typename ContainerT>
    ContainerT calculate_error(
      const Field h,
      const Coefficients::KCoefficients<S, ContainerT>& k_coefficients)
    {
      ContainerT y_err {
        delta_coefficients_.get_ith_element(1) *
          k_coefficients.get_ith_coefficient(1)};

      for (std::size_t j {2}; j <= S; ++j)
      {
        y_err += delta_coefficients_.get_ith_element(j) *
          k_coefficients.get_ith_coefficient(j);
      }

      // h * \sum_{j=1}^s (b_j - b^*_j) k_j
      return h * y_err;
    }

  protected:

    //--------------------------------------------------------------------------
    /// \tparam N Dimension of out or of Container T, or of y in the ODE.
    //--------------------------------------------------------------------------
    template <typename ContainerT, std::size_t N>
    void sum_a_and_k_products(
      Coefficients::KCoefficients<S, ContainerT>& k_coefficients,
      const std::size_t l,
      const Field h,
      ContainerT& out)
    {
      std::array<Field, N> a_lj_times_k_j {};

      k_coefficients.scalar_multiply(a_lj_times_k_j, 1, get_a_ij(l, 1));

      std::copy(a_lj_times_k_j.begin(), a_lj_times_k_j.end(), out.begin());

      for (std::size_t j {2}; j < l; ++j)
      {
        // Calculate a_lj * k_j
        k_coefficients.scalar_multiply(a_lj_times_k_j, j, get_a_ij(l, j));

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
