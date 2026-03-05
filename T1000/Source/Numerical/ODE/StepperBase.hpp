//------------------------------------------------------------------------------
/// \file StepperBase.hpp
/// \brief Base type for all Runge-Kutta steppers (Numerical Recipes §17.2
///        design, modernised to C++20).
///
/// ## What a Stepper IS (interview concept)
///
/// A "stepper" encapsulates one trial step of an ODE integrator.  It owns the
/// current integration state and exposes two operations:
///
///   do_step(h, derivs)   – advance by exactly h (no error control).
///   step(htry, derivs)   – adaptive: try h = htry, reject+retry if error
///                          too large, fill hdid / hnext on exit.
///
/// Every adaptive stepper maintains:
///   y      – state at current x  (updated when step accepted)
///   dydx   – derivative at current x
///   yout   – proposed new state after do_step
///   yerr   – error estimate  (yout_high - yout_low for embedded pairs)
///   hdid   – step size that was actually accepted
///   hnext  – recommended next step size
///
/// ## Error norm (NR §17.2.1)
///
/// Mixed absolute/relative tolerance avoids the two failure modes:
///   - Pure absolute: struggles near zero (tiny signal, big relative error)
///   - Pure relative: struggles far from zero (huge scale, meaningless atol)
///
/// Scale for each component i:
///   sc_i = atol + rtol * max(|y_i|, |yout_i|)
///
/// Scaled RMS error norm:
///   err = sqrt( (1/N) * Σ (yerr_i / sc_i)^2 )
///
/// Step accepted when err ≤ 1.
//------------------------------------------------------------------------------
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace Numerical {
namespace ODE {

/// \tparam N  Dimension of the state vector (compile-time constant).
template<std::size_t N>
struct StepperBase
{
  using State = std::array<double, N>;

  // ── Integration state ─────────────────────────────────────────────────────
  double x    {0.0};   ///< Current value of the independent variable (time).
  double xold {0.0};   ///< x at the start of the last accepted step.
  State  y    {};      ///< State vector at x.
  State  dydx {};      ///< dy/dx evaluated at (x, y).
  State  yout {};      ///< Proposed new state after do_step.
  State  yerr {};      ///< Per-component error estimate.

  // ── Step-size bookkeeping ─────────────────────────────────────────────────
  double hdid  {0.0};  ///< Step size actually taken (set by step()).
  double hnext {0.0};  ///< Recommended next step size (set by step()).

  // ── Tolerances ────────────────────────────────────────────────────────────
  double atol {1.0e-6}; ///< Absolute tolerance (floor on sc_i).
  double rtol {1.0e-6}; ///< Relative tolerance.

  // ── Constructor ───────────────────────────────────────────────────────────
  StepperBase() = default;
  StepperBase(const State& y0, double x0,
              double absolute_tol = 1.0e-6,
              double relative_tol = 1.0e-6)
    : x{x0}, y{y0}, atol{absolute_tol}, rtol{relative_tol}
  {}

  // ── Error norm ────────────────────────────────────────────────────────────
  /// Compute the scaled RMS error norm.  Returns a value ≤ 1 when the step
  /// satisfies the requested tolerances.
  double error_norm() const
  {
    double sum = 0.0;
    for (std::size_t i = 0; i < N; ++i)
    {
      const double sc = atol + rtol * std::max(std::abs(y[i]),
                                               std::abs(yout[i]));
      const double ratio = yerr[i] / sc;
      sum += ratio * ratio;
    }
    return std::sqrt(sum / static_cast<double>(N));
  }
};

} // namespace ODE
} // namespace Numerical
