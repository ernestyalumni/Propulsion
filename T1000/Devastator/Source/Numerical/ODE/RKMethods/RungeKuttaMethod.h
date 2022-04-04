#ifndef NUMERICAL_ODE_RK_METHODS_RUNGE_KUTTA_METHOD_H
#define NUMERICAL_ODE_RK_METHODS_RUNGE_KUTTA_METHOD_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <stdexcept>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

//------------------------------------------------------------------------------
/// \tparam M - number of stages.
/// \tparam N - dimension of the dependent variable.
//------------------------------------------------------------------------------
template <
  std::size_t M,
  typename Field = double
  >
class CalculateNextStep
{
  public:

    CalculateNextStep() = delete;
    CalculateNextStep(
      const std::initializer_list<Field>& alpha_coefficients,
      const std::initializer_list<Field>& beta_coefficients,
      const std::initializer_list<Field>& c_coefficients)
    {
      if (alpha_coefficients.size() != M - 1)
      {
        throw std::runtime_error("Wrong size alpha inputs");
      }

      if (beta_coefficients.size() != M * (M - 1) / 2)
      {
        throw std::runtime_error("Wrong size beta inputs");
      }

      if (c_coefficients.size() != M)
      {
        throw std::runtime_error("Wrong size c inputs");
      }

      std::copy(
        beta_coefficients.begin(),
        beta_coefficients.end(),
        beta_coefficients_.begin());

      std::copy(
        c_coefficients.begin(),
        c_coefficients.end(),
        c_coefficients_.begin());

      std::copy(
        alpha_coefficients.begin(),
        alpha_coefficients.end(),
        alpha_coefficients_.begin());
    }

    virtual ~CalculateNextStep() = default;

    //--------------------------------------------------------------------------
    /// \param h - step size. h = (b - a) / M_segments
    //--------------------------------------------------------------------------    
    template <typename ContainerT, typename DerivativeT>
    ContainerT operator()(
      const ContainerT& x_n,
      const Field t_n,
      const Field h)
    {
      std::array<ContainerT, M> k_coefficients {
        calculate_k_coefficients(x_n, t_n, h)};

      ContainerT summation;

      std::transform(
        k_coefficients[0].begin(),
        k_coefficients[0].end(),
        summation.begin(),
        std::bind(
          std::multiplies<Field>(),
          std::placeholders::_1,
          get_c_i(1)));

      for (std::size_t i {2}; i <= M; ++i)
      {
        ContainerT summand;

        std::transform(
          k_coefficients[i - 1].begin(),
          k_coefficients[i - 1].end(),
          summand.begin(),
          std::bind(
            std::multiplies<Field>(),
            std::placeholders::_1,
            get_c_i(i)));

        std::transform(
          summation.begin(),
          summation.end(),
          summand.begin(),
          summation.begin(),
          std::plus<Field>());
      }

      std::transform(
        summation.begin(),
        summation.end(),
        summation.begin(),
        std::bind(
          std::multiplies<Field>(),
          std::placeholders::_1,
          h));

      std::transform(
        summation.begin(),
        summation.end(),
        x_n.begin(),
        summation.begin(),
        std::plus<Field>());

      return summation;
    }


    Field get_beta_ij(const std::size_t i, const std::size_t j) const
    {
      assert(i > j && j >= 1 && i <= M);

      std::size_t n {i - 2};
      n = n * (n + 1) / 2;

      return beta_coefficients_[n + (j - 1)];
    }

    Field get_alpha_i(const std::size_t i) const
    {
      assert(i >= 2 && i <= M);

      return alpha_coefficients_[i - 2];      
    }

    Field get_c_i(const std::size_t i) const
    {
      assert(i >= 1 && i <= M);

      return c_coefficients_[i - 1];
    }

  protected:

    template <typename ContainerT, typename DerivativeT>
    std::array<ContainerT, M> calculate_k_coefficients(
      const ContainerT& x_n,
      const Field t_n,
      const Field h)
    {
      DerivativeT f;

      std::array<ContainerT, M> k_coefficients;
      k_coefficients[0] = f(t_n, x_n);

      for (std::size_t l {2}; l <= M; ++l)
      {
        const Field t_l {t_n + get_alpha_i(l)};

        ContainerT x_l {
          sum_beta_and_k_products<ContainerT>(k_coefficients, l, h)};

        std::transform(
          x_l.begin(),
          x_l.end(),
          x_n.begin(),
          x_l.begin(),
          std::plus<Field>());

        k_coefficients[l - 1] = f(t_l, x_l);
      }

      return k_coefficients;
    }

    //--------------------------------------------------------------------------
    /// \param l - l = 2...M, sum with up to l - 1 k coefficients.
    //--------------------------------------------------------------------------       

    template <typename ContainerT>
    ContainerT sum_beta_and_k_products(
      std::array<ContainerT, M>& k_coefficients,
      const std::size_t l,
      const Field h)
    {
      assert(l <= M && l >= 2);

      // j = 1...l-1
      ContainerT beta_lj_times_k_j;

      std::transform(
        k_coefficients[0].begin(),
        k_coefficients[0].end(),
        beta_lj_times_k_j.begin(),
        std::bind(
          std::multiplies<Field>(),
          std::placeholders::_1,
          get_beta_ij(l, 1)));

      ContainerT summation {beta_lj_times_k_j};

      for (std::size_t j {2}; j < l - 1; ++j)
      {
        std::transform(
          k_coefficients[j - 1].begin(),
          k_coefficients[j - 1].end(),
          beta_lj_times_k_j.begin(),
          std::bind(
            std::multiplies<Field>(),
            std::placeholders::_1,
            get_beta_ij(l, j)));

        std::transform(
          summation.begin(),
          summation.end(),
          beta_lj_times_k_j.begin(),
          summation.begin(),
          std::plus<Field>());
      }

      std::transform(
        summation.begin(),
        summation.end(),
        summation.begin(),
        std::bind(
          std::multiplies<Field>(),
          std::placeholders::_1,
          h));

      return summation;
    }

  private:

    std::array<Field, M * (M - 1) / 2> beta_coefficients_;
    std::array<Field, M> c_coefficients_;
    std::array<Field, M - 1> alpha_coefficients_; 
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_RUNGE_KUTTA_METHOD_H
