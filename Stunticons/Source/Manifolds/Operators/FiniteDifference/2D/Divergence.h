#ifndef MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_2D_DIVERGENCE_H
#define MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_2D_DIVERGENCE_H

namespace Manifolds
{
namespace Operators
{
namespace FiniteDifference
{

template <typename FPT, typename CompoundFPT, int NU>
__device__ FPT divergence(CompoundFPT stencil[NU][2])
{
  FPT stencil_x[NU]
}

} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds

#endif // MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_2D_DIVERGENCE_H