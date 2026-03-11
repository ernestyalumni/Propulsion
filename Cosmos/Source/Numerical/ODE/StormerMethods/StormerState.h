#ifndef NUMERICAL_ODE_STORMER_METHODS_STORMER_STATE_H
#define NUMERICAL_ODE_STORMER_METHODS_STORMER_STATE_H

#include <utility> // std::move

namespace Numerical
{
namespace ODE
{
namespace StormerMethods
{

//------------------------------------------------------------------------------
/// \brief Position-velocity pair representing the state for a second-order ODE
///   y'' = f(x, y).
///
/// \details For a first-order system y' = F(x, y), the full state is a single
///   vector. For a second-order system, position and velocity are distinct and
///   must be tracked separately because f does NOT depend on velocity — only
///   position is passed to the derivative functor.
///
/// \tparam ContainerT - type for both position and velocity (e.g. std::valarray,
///   NVector<N>, std::vector).
//------------------------------------------------------------------------------
template <typename ContainerT>
struct StormerState
{
  ContainerT position;
  ContainerT velocity;

  StormerState() = default;

  StormerState(const ContainerT& q, const ContainerT& v):
    position{q},
    velocity{v}
  {}

  StormerState(ContainerT&& q, ContainerT&& v):
    position{std::move(q)},
    velocity{std::move(v)}
  {}
};

} // namespace StormerMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_STORMER_METHODS_STORMER_STATE_H
