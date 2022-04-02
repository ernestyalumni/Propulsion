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
namespace DOPRI5Coefficients
{

extern const Coefficients::ACoefficients<7, double> a_coefficients;

} // namespace Coefficients  
} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_DOPRI5_COEFFICIENTS_H
