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

template <typename FPT>
__device__ FPT directional_derivative_1(FPT stencil[1][2], FPT c_nus[4])
{
  FPT return_value {static_cast<FPT>(0)};

  return_value += c_nus[0] * (stencil[0][0] - stencil[0][1]);

  return return_value;
}

template <typename FPT>
__device__ FPT directional_derivative_2(FPT stencil[2][2], FPT c_nus[4])
{
  static constexpr int NU {2};

  FPT return_value {static_cast<FPT>(0)};

  for (int nu {0}; nu < NU; ++nu)
  {
    return_value += c_nus[nu] * (stencil[nu][1] - stencil[nu][0]);
  }

  return return_value;
}

template <typename FPT>
__device__ FPT directional_derivative_3(FPT stencil[3][2], FPT c_nus[4])
{
  static constexpr int NU {3};

  FPT return_value {static_cast<FPT>(0)};

  for (int nu {0}; nu < NU; ++nu)
  {
    return_value += c_nus[nu] * (stencil[nu][1] - stencil[nu][0]);
  }

  return return_value;
}

template <typename FPT>
__device__ FPT directional_derivative_4(FPT stencil[4][2], FPT c_nus[4])
{
  static constexpr int NU {4};

  FPT return_value {static_cast<FPT>(0)};

  for (int nu {0}; nu < NU; ++nu)
  {
    return_value += c_nus[nu] * (stencil[nu][1] - stencil[nu][0]);
  }

  return return_value;
}

template <typename FPT, int NU>
__device__ FPT directional_derivative(FPT stencil[NU][2], FPT c_nus[NU])
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
