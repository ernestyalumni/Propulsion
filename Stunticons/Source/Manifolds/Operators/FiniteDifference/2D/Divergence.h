#ifndef MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_2D_DIVERGENCE_H
#define MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_2D_DIVERGENCE_H

#include "Manifolds/Operators/FiniteDifference/C_Nu_Coefficients.h"
#include "Manifolds/Operators/FiniteDifference/2D/DirectionalDerivatives.h"

namespace Manifolds
{
namespace Operators
{
namespace FiniteDifference
{
namespace TwoDimensional
{

template <typename FPT, typename CompoundFPT, int NU>
__device__ FPT divergence(CompoundFPT stencil[NU][2])
{
  FPT stencil_x[NU][2] {};
  FPT stencil_y[NU][2] {};

  for (int i {0}; i < NU; ++i)
  {
    for (int j {0}; j < 2; ++j)
    {
      stencil_x[i][j] = stencil[i][j].x;
      stencil_y[i][j] = stencil[i][j].y;
    }
  }

  FPT c_nus_x[4] {
    cnu_coefficients_first_order<CompoundFPT>[0].x,
    cnu_coefficients_first_order<CompoundFPT>[1].x,
    cnu_coefficients_first_order<CompoundFPT>[2].x,
    cnu_coefficients_first_order<CompoundFPT>[3].x
  };

  FPT c_nus_y[4] {
    cnu_coefficients_first_order<CompoundFPT>[0].y,
    cnu_coefficients_first_order<CompoundFPT>[1].y,
    cnu_coefficients_first_order<CompoundFPT>[2].y,
    cnu_coefficients_first_order<CompoundFPT>[3].y
  };

  FPT divergence_x {directional_derivative<FPT, NU>(stencil_x, c_nus_x)};
  FPT divergence_y {directional_derivative<FPT, NU>(stencil_y, c_nus_y)};

  return divergence_x + divergence_y;
}

} // namespace TwoDimensional
} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds

#endif // MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_2D_DIVERGENCE_H