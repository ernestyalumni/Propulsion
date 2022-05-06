#ifndef NUMERICAL_ODE_RK_METHODS_CALCULATE_HERMITE_INTERPOLATION_H
#define NUMERICAL_ODE_RK_METHODS_CALCULATE_HERMITE_INTERPOLATION_H

#include "Algebra/Modules/Vectors/NVector.h"

#include <cstddef>
#include <valarray>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

template <typename Field = double>
std::valarray<Field> calculate_hermite_interpolation(
  const std::valarray<Field>& y_0,
  const std::valarray<Field>& y_1,
  const std::valarray<Field>& dydx_0,
  const std::valarray<Field>& dydx_1,
  const Field theta,
  const Field h)
{
  return ((1. - 3. * theta * theta + 2. * theta * theta * theta) * y_0) +
    ((3. * theta * theta - 2. * theta * theta * theta) * y_1) +
    (theta * (theta - 1)) * (
      (((theta - 1.) * h) * dydx_0) +
      ((theta * h) * dydx_1));
}

template <std::size_t N, typename Field = double>
Algebra::Modules::Vectors::NVector<N, Field> calculate_hermite_interpolation(
  const Algebra::Modules::Vectors::NVector<N, Field>& y_0,
  const Algebra::Modules::Vectors::NVector<N, Field>& y_1,
  const Algebra::Modules::Vectors::NVector<N, Field>& dydx_0,
  const Algebra::Modules::Vectors::NVector<N, Field>& dydx_1,
  const Field theta,
  const Field h)
{
  return ((1. - 3. * theta * theta + 2. * theta * theta * theta) * y_0) +
    ((3. * theta * theta - 2. * theta * theta * theta) * y_1) +
    (theta * (theta - 1)) * (
      (((theta - 1.) * h) * dydx_0) +
      ((theta * h) * dydx_1));
}

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_CALCULATE_HERMITE_INTERPOLATION_H
