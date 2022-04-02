#ifndef NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_DOPRI5_COEFFICIENTS_H
#define NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_DOPRI5_COEFFICIENTS_H

#include "ACoefficients.h"

#include <array>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{
namespace Coefficients
{

template <typename Field>
extern const ACoefficients<7, Field> a_coefficients {
  0.2,
  3.0 / 40.0
};

} // namespace Coefficients  
} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_DOPRI5_COEFFICIENTS_H
