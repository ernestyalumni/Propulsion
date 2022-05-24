#ifndef GOOGLE_UNIT_TESTS_NUMERICAL_ODE_RK_METHODS_TEST_SETUP_H
#define GOOGLE_UNIT_TESTS_NUMERICAL_ODE_RK_METHODS_TEST_SETUP_H

#include "Algebra/Modules/Vectors/NVector.h"
#include "Numerical/ODE/RKMethods/Coefficients/DOPRI5Coefficients.h"
#include "Numerical/ODE/RKMethods/Coefficients/KCoefficients.h"
#include "Numerical/ODE/RKMethods/StepInputs.h"

#include <valarray>

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

inline constexpr double epsilon {1.0e-6};

// k = 5 for 5th order for O(h^5)
inline constexpr double alpha_5 {0.7 / 5.0};
inline constexpr double beta_5 {0.08};

inline constexpr size_t DOPRI5_s {
  ::Numerical::ODE::RKMethods::DOPRI5Coefficients::s};

inline const auto& DOPRI5_a_coefficients =
  ::Numerical::ODE::RKMethods::DOPRI5Coefficients::a_coefficients;

inline const auto& DOPRI5_c_coefficients =
  ::Numerical::ODE::RKMethods::DOPRI5Coefficients::c_coefficients;

inline const auto& DOPRI5_delta_coefficients =
  ::Numerical::ODE::RKMethods::DOPRI5Coefficients::delta_coefficients;

inline auto example_f_with_std_valarray = [](
  const double x,
  std::valarray<double>& y)
{
  return y - x * x + 1.0;
};

template <std::size_t N>
auto example_f_with_NVector = [](
  const double x,
  const Algebra::Modules::Vectors::NVector<N>& y)
{
  return y - x * x + 1.0;
};

inline auto exact_solution = [](const double x)
{
  return x * x + 2.0 * x + 1.0 - 0.5 * std::exp(x);
};

template <size_t S>
struct ExampleSetupWithStdValarray
{
  double h_ {0.5};
  double x_n_ {0.0};
  const double t_0_ {0.0};
  const double t_f_ {2.0};
  const std::valarray<double> y_0_ {0.5};
  const std::valarray<double> dydx_0_ {1.5};

  std::valarray<double> y_n_ {0.5};
  std::valarray<double> y_out_ {0.0};
  std::valarray<double> y_err_ {0.0};

  ::Numerical::ODE::RKMethods::Coefficients::KCoefficients<
    S,
    std::valarray<double>> k_coefficients_;

  double previous_error_ {1.0e-4};

  ExampleSetupWithStdValarray()
  {
    k_coefficients_[0] = dydx_0_;
  }
};

template <size_t S, size_t N>
struct ExampleSetupWithNVector
{
  using NVector = Algebra::Modules::Vectors::NVector<N>;

  double h_ {0.5};
  double x_n_ {0.0};
  const double t_0_ {0.0};
  const double t_f_ {2.0};
  const NVector y_0_ {0.5};
  const NVector dydx_0_ {1.5};

  NVector y_n_ {0.5};
  NVector y_out_ {0.0};
  NVector y_err_ {0.0};

  ::Numerical::ODE::RKMethods::Coefficients::KCoefficients<
    S,
    NVector> k_coefficients_;

  double previous_error_ {1.0e-4};

  ExampleSetupWithNVector()
  {
    k_coefficients_[0] = dydx_0_;
  }
};

inline ::Numerical::ODE::RKMethods::StepInputs<DOPRI5_s, std::valarray<double>>
  step_inputs_with_std_valarray {
    std::valarray<double>{0.5},
    std::valarray<double>{1.5},
    0.5,
    0.0};

inline ::Numerical::ODE::RKMethods::StepInputs<
  DOPRI5_s,
  Algebra::Modules::Vectors::NVector<1>>
    step_inputs_with_nvector {
      Algebra::Modules::Vectors::NVector<1>{0.5},
      Algebra::Modules::Vectors::NVector<1>{1.5},
      0.5,
      0.0};

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests

#endif // GOOGLE_UNIT_TESTS_NUMERICAL_ODE_RK_METHODS_TEST_SETUP_H