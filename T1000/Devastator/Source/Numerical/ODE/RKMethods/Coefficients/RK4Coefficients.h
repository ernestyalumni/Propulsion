#ifndef NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_RK4_COEFFICIENTS_H
#define NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_RK4_COEFFICIENTS_H

#include "ACoefficients.h"
#include "BCoefficients.h"
#include "CCoefficients.h"

#include <array>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{
namespace RK4Coefficients
{

//------------------------------------------------------------------------------
/// \brief Stage value.
/// \ref Organizing static data in C++
/// \url https://stackoverflow.com/questions/7535743/organizing-static-data-in-c
//------------------------------------------------------------------------------
static constexpr std::size_t s {4};

extern const Coefficients::ACoefficients<s> a_coefficients;
extern const Coefficients::BCoefficients<s> b_coefficients;
extern const Coefficients::CCoefficients<s> c_coefficients;

} // namespace RK4Coefficients  
} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_RK4_COEFFICIENTS_H
