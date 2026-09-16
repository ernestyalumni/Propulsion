//------------------------------------------------------------------------------
/// \file Propagator.hpp
/// \brief High-level orbit propagator: integrates the two-body EOM from t0
///        to tf, recording trajectory and conserved quantities.
///
/// Usage example:
/// \code
///   using namespace OrbitalMechanics;
///   KeplerianElements el{7e6, 0.01, 0.5, 1.0, 0.5, 0.0};
///   StateVector6 y0 = elements_to_state(el);
///
///   PropagatorResult result = propagate(y0, 0.0, orbital_period(el.a));
///   // result.states[i]   — state at t_i
///   // result.times[i]    — t_i
/// \endcode
//------------------------------------------------------------------------------
#pragma once

#include "TwoBody.hpp"
#include "OrbitalElements.hpp"
#include "StateVector.hpp"
#include "Numerical/ODE/RKMethods/IntegrateWithPIControl.h"
#include "Numerical/ODE/RKMethods/Coefficients/DOPRI5Coefficients.h"

#include <vector>
#include <cmath>

namespace OrbitalMechanics {

// ── Result container ─────────────────────────────────────────────────────────

struct PropagatorResult
{
  std::vector<double>       times;   ///< Epoch of each recorded point [s].
  std::vector<StateVector6> states;  ///< Cartesian state at each epoch.
  std::vector<double>       energies;///< Specific energy ε = v²/2 − μ/r [J/kg].
  std::vector<double>       hmags;   ///< |h| = |r×v| [m²/s].
  std::size_t               n_steps {0}; ///< Total accepted integration steps.
};

// ── Observer that records the trajectory ─────────────────────────────────────

struct TrajectoryObserver
{
  PropagatorResult& result;
  double            mu;

  void operator()(double t, const StateVector6& y)
  {
    result.times.push_back(t);
    result.states.push_back(y);
    result.energies.push_back(specific_energy(y, mu));
    result.hmags.push_back(angular_momentum_mag(y));
  }
};

// ── Main propagation function ─────────────────────────────────────────────────

/// \brief Propagate a satellite from t0 to tf using DOPRI5 adaptive stepper.
///
/// \param y0       Initial state vector [m, m/s] in ECI.
/// \param t0       Start time [s].
/// \param tf       End time [s].
/// \param mu       Gravitational parameter [m³/s²].
/// \param atol     Absolute tolerance (position/velocity components).
/// \param rtol     Relative tolerance.
/// \param h_init   Initial step-size hint [s]  (default: period/100 or span/100).
/// \param record   If true, record state at every accepted step.
/// \return PropagatorResult with trajectory data.
inline PropagatorResult propagate(
  const StateVector6& y0,
  double t0,
  double tf,
  double mu     = Constants::MU_EARTH,
  double atol   = 1.0e-6,
  double rtol   = 1.0e-8,
  double h_init = 0.0,
  bool   record = true)
{
  PropagatorResult result;

  // Default initial step: span / 100
  if (h_init <= 0.0)
    h_init = std::abs(tf - t0) / 100.0;

  namespace RK = Numerical::ODE::RKMethods;
  namespace DP = RK::DOPRI5Coefficients;
  // Cosmos owns the tableau, step calculation, PI controller and driver.
  // std::array<double, 6> preserves the application's fixed state storage.
  RK::IntegrateWithPIControl integrate {
    RK::CalculateNewYAndError<DP::s, TwoBodyEOM>{
      TwoBodyEOM{mu}, DP::a_coefficients, DP::c_coefficients, DP::delta_coefficients},
    RK::CalculateScaledError{atol, rtol},
    RK::ComputePIStepSize{0.7/5.0, 0.4/5.0},
    1000000};

  // Observer
  TrajectoryObserver obs{result, mu};

  // Record initial state
  if (record)
  {
    result.times.push_back(t0);
    result.states.push_back(y0);
    result.energies.push_back(specific_energy(y0, mu));
    result.hmags.push_back(angular_momentum_mag(y0));
  }

  const auto [final_time, final_state, steps] = integrate.integrate_with_observer<6>(
    RK::IntegrationInputs{y0, t0, tf, h_init},
    [&](double t, const StateVector6& y, double) { if (record) obs(t, y); },
    100, 1.0e-12);
  result.n_steps = steps;
  if (!record)
  {
    result.times.push_back(final_time);
    result.states.push_back(final_state);
    result.energies.push_back(specific_energy(final_state, mu));
    result.hmags.push_back(angular_momentum_mag(final_state));
  }

  return result;
}

} // namespace OrbitalMechanics
