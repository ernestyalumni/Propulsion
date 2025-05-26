#ifndef MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_DIRECTIONAL_DERIVATIVES_H
#define MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_DIRECTIONAL_DERIVATIVES_H

namespace Manifolds
{
namespace Operators
{
namespace FiniteDifference
{

//------------------------------------------------------------------------------
/// \param FPT The floating point type.
//------------------------------------------------------------------------------

template <typename FPT, int NU>
__device__ FPT directional_derivative(FPT stencil[NU][2], FPT c_nus[4])
{
  FPT return_value {static_cast<FPT>(0)};

  for (int nu {0}; nu < NU; ++nu)
  {
    return_value += c_nus[nu] * (stencil[nu][1] - stencil[nu][0]);
  }

  return return_value;
}

} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds

#endif // MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_DIRECTIONAL_DERIVATIVES_H
