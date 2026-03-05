//------------------------------------------------------------------------------
/// \file TwoBody.hpp
/// \brief Two-body gravitational equation of motion.
///
/// ## Physics (interview)
///
/// ### The two-body problem
/// Two point masses m₁ and m₂.  Their relative position r = r₁ − r₂ obeys:
///
///   r̈ = −μ/|r|³ · r        (Cowell's formulation in Cartesian ECI)
///
/// where μ = G(m₁+m₂).  For Earth-satellite: μ ≈ G·M_Earth because m_sat ≪ M_Earth.
///
/// ### State vector form (for ODE integrator)
/// Let y = [x, y, z, vx, vy, vz].  Then:
///
///   ẏ = [ vx, vy, vz,  −μx/r³,  −μy/r³,  −μz/r³ ]
///
/// This is a 1st-order system of 6 ODEs — fed directly to RKCK or DOPRI5.
///
/// ### Conserved quantities (use to verify integration accuracy)
///   Energy:           ε  = v²/2 − μ/r              (should stay constant)
///   Angular momentum: |h| = |r × v|                 (should stay constant)
///   Eccentricity vec: e   = v×h/μ − r̂              (direction/magnitude constant)
///
/// ### Perturbations (extensions)
/// Real missions add perturbation accelerations:
///   - J2 (Earth oblateness): dominant non-spherical term
///   - Atmospheric drag: a = −½ρ(CdA/m)v²v̂
///   - Third-body gravity: Moon, Sun
///   - Solar radiation pressure
/// These are simply added to the acceleration terms in the EOM.
///
/// ### Cowell vs Encke
///   Cowell:  integrate r directly.  Simple; accumulates error on long arcs.
///   Encke:   integrate Δr = r − r_ref (deviation from reference conic).
///            More accurate for near-Keplerian orbits; reset periodically.
//------------------------------------------------------------------------------
#pragma once

#include "StateVector.hpp"
#include "Constants.hpp"
#include <cmath>
#include <stdexcept>

namespace OrbitalMechanics {

/// \brief Two-body point-mass equation of motion.
///
/// Call operator satisfies the Derivs concept:
///   StateVector6 d = eom(t, y)
///
/// \note t is unused in the pure two-body problem (autonomous system),
///       but kept for compatibility with the ODE stepper interface.
struct TwoBodyEOM
{
  double mu; ///< Standard gravitational parameter [m³/s²].

  explicit TwoBodyEOM(double gravitational_parameter = Constants::MU_EARTH)
    : mu{gravitational_parameter}
  {}

  StateVector6 operator()(double /*t*/, const StateVector6& y) const
  {
    const double r = radius(y);
    if (r < 1.0)
      throw std::domain_error("TwoBodyEOM: radius < 1 m — singularity");

    const double r3 = r * r * r;
    const double c  = -mu / r3;

    return StateVector6{
      y[3],       // ẋ  = vx
      y[4],       // ẏ  = vy
      y[5],       // ż  = vz
      c * y[0],   // v̇x = −μ x / r³
      c * y[1],   // v̇y = −μ y / r³
      c * y[2]    // v̇z = −μ z / r³
    };
  }
};

// ── Convenience: expected orbital period ────────────────────────────────────

/// Keplerian period T = 2π√(a³/μ).
inline double orbital_period(double semi_major_axis, double mu = Constants::MU_EARTH)
{
  return Constants::TWO_PI * std::sqrt(
    semi_major_axis * semi_major_axis * semi_major_axis / mu);
}

/// Circular orbit radius for a given period.
inline double circular_radius_from_period(double T,
                                          double mu = Constants::MU_EARTH)
{
  return std::cbrt(mu * (T / Constants::TWO_PI) * (T / Constants::TWO_PI));
}

/// Circular orbit speed at radius r.
inline double circular_speed(double r, double mu = Constants::MU_EARTH)
{
  return std::sqrt(mu / r);
}

} // namespace OrbitalMechanics
