//------------------------------------------------------------------------------
/// \file ODEDriver.hpp
/// \brief Adaptive ODE integration driver (Numerical Recipes §17.2.2 "odeint").
///
/// ## Interview concepts
///
/// integrate_adaptive drives a Stepper from x1 to x2 by repeatedly calling
/// stepper.step(h, derivs).  Key responsibilities:
///
///   1. Initialise step size h (user hint or auto).
///   2. After each accepted step, advance x by hdid and use hnext.
///   3. Final step: clamp h so we land exactly on x2.
///   4. Optionally record (x, y) pairs in a user-supplied observer.
///   5. Guard against step-size underflow and infinite loops.
///
/// ### Observer pattern
/// The optional Observer callback is called after every accepted step:
///   observer(stepper.x, stepper.y)
/// Used to record trajectories, detect events, or print progress.
///
/// ### Why not just use fixed-step RK4?
/// Adaptive steppers achieve a given accuracy with far fewer f-evaluations:
///   - Fixed RK4 on a circular orbit (smooth): fine.
///   - Fixed RK4 near periapsis of a highly elliptical orbit: catastrophic
///     (large curvature → need tiny h exactly where you can't predict it).
///   Adaptive: step sizes shrink automatically near periapsis and grow near
///   apoapsis, keeping error per step roughly constant.
//------------------------------------------------------------------------------
#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>

namespace Numerical {
namespace ODE {

/// \brief No-op observer (default: record nothing).
struct NullObserver
{
  template<typename State>
  void operator()(double /*x*/, const State& /*y*/) const noexcept {}
};

/// \brief Integrate from x1 to x2 using an adaptive stepper.
///
/// \tparam Stepper   Must have: step(double h, Derivs&), members x, y, hdid, hnext.
/// \tparam Derivs    Callable: State f(double t, const State& y).
/// \tparam Observer  Callable: void obs(double t, const State& y).
///
/// \param stepper    Stepper initialised with y0 at x1.
/// \param derivs     RHS function of the ODE.
/// \param x1         Start of integration interval.
/// \param x2         End of integration interval.
/// \param h_init     Initial step-size hint (sign must match x2-x1).
/// \param h_min      Minimum allowed step size (throws on underflow).
/// \param max_steps  Safety cap on total steps (throws if exceeded).
/// \param observer   Called after every accepted step.
///
/// \return  Number of accepted steps taken.
template<typename Stepper, typename Derivs,
         typename Observer = NullObserver>
std::size_t integrate_adaptive(
  Stepper&  stepper,
  Derivs&   derivs,
  double    x1,
  double    x2,
  double    h_init,
  double    h_min      = 1.0e-14,
  std::size_t max_steps = 100000,
  Observer  observer   = Observer{})
{
  if (x2 == x1) return 0;

  // Initialise stepper position
  stepper.x = x1;

  double h = h_init;
  if ((x2 - x1) * h < 0.0)
    h = -h;   // ensure sign matches direction

  std::size_t nstep = 0;

  while (true)
  {
    if (nstep >= max_steps)
      throw std::runtime_error(
        "integrate_adaptive: max_steps exceeded — possible stiffness or too-tight tolerance");

    // Clamp to not overshoot x2
    if ((stepper.x + h - x2) * (stepper.x + h - x1) > 0.0)
      h = x2 - stepper.x;

    stepper.step(h, derivs);
    ++nstep;

    observer(stepper.x, stepper.y);

    if ((stepper.x - x2) * (x2 - x1) >= 0.0)
      break;   // reached or passed x2

    if (std::abs(stepper.hnext) < h_min)
      throw std::runtime_error(
        "integrate_adaptive: step size underflow — check tolerances or problem stiffness");

    h = stepper.hnext;
  }

  return nstep;
}

} // namespace ODE
} // namespace Numerical
