#ifndef NUMERICAL_ODE_STORMER_METHODS_NUMEROV_STATE_H
#define NUMERICAL_ODE_STORMER_METHODS_NUMEROV_STATE_H

#include <utility> // std::move

namespace Numerical
{
namespace ODE
{
namespace StormerMethods
{

//------------------------------------------------------------------------------
/// \brief State for Numerov's method applied to y'' = f(x, y).
///
/// \details Numerov's method is a two-step (k=2) multistep method:
///
///   y_{n+1} - 2*y_n + y_{n-1} = h^2/12 * (f_{n+1} + 10*f_n + f_{n-1})
///
/// Unlike StormerState (one-step, holds only current position + velocity),
/// NumerovState carries the HISTORY needed for the three-point recurrence:
///
///   - position  : y_n   (current, ORDER 4 accurate in h)
///   - velocity  : v_n   (current, ORDER 2 via leapfrog formula — see note)
///   - y_prev    : y_{n-1}  (previous position, required for next step)
///   - f_prev    : f_{n-1}  (previous force, required for corrector)
///
/// Accuracy note:
///   Position accuracy is O(h^4) (global, from Numerov's method).
///   Velocity accuracy is O(h^2) (from the leapfrog half-kick formula
///   v_n = (y_n - y_{n-1})/h + h/2 * f_n).  If O(h^4) velocity is needed,
///   use central differences: v_n ~ (y_{n+1} - y_{n-1}) / (2h) after the
///   next step is available, or fit a cubic polynomial to positions.
///
/// Starting procedure:
///   Numerov requires TWO starting values (y_0 and y_1) to launch.
///   For global order-4 convergence (HNW Theorem 10.6, p. 468), y_1 must
///   satisfy |y(x_1) - y_1| = O(h^5).  Use NumerovStep::startup() which
///   calls StormerStep with many sub-steps for an accurate y_1, or supply
///   a 4th-order Runge-Kutta-Nyström starting step externally.
///
/// Reference: Hairer, Nørsett, Wanner, "Solving ODEs I" §III.10, pp. 461-473.
///
/// \tparam ContainerT - type for position, velocity vectors
///   (e.g. std::valarray<double>, NVector<N>).
//------------------------------------------------------------------------------
template <typename ContainerT>
struct NumerovState
{
  ContainerT position;  ///< y_n  — current position, O(h^4) global accuracy
  ContainerT velocity;  ///< v_n  — current velocity, O(h^2) global accuracy
  ContainerT y_prev;    ///< y_{n-1} — needed for the 3-point LHS stencil
  ContainerT f_prev;    ///< f_{n-1} — needed for the implicit corrector RHS

  NumerovState() = default;

  NumerovState(
    const ContainerT& pos,
    const ContainerT& vel,
    const ContainerT& y_p,
    const ContainerT& f_p)
    : position{pos}
    , velocity{vel}
    , y_prev{y_p}
    , f_prev{f_p}
  {}

  NumerovState(
    ContainerT&& pos,
    ContainerT&& vel,
    ContainerT&& y_p,
    ContainerT&& f_p)
    : position{std::move(pos)}
    , velocity{std::move(vel)}
    , y_prev{std::move(y_p)}
    , f_prev{std::move(f_p)}
  {}
};

} // namespace StormerMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_STORMER_METHODS_NUMEROV_STATE_H
