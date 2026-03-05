//------------------------------------------------------------------------------
/// \file OrbitalElements.hpp
/// \brief Keplerian orbital elements ↔ Cartesian state vector conversions.
///
/// ## Interview concepts
///
/// ### The 6 classical Keplerian elements
///
///   a  – semi-major axis      [m]       (size of orbit)
///   e  – eccentricity         [-]       (shape; 0=circle, <1=ellipse,
///                                        1=parabola, >1=hyperbola)
///   i  – inclination          [rad]     (tilt of orbital plane vs equatorial)
///   Ω  – RAAN (right ascension of ascending node) [rad]  (rotation of orbital
///           plane around Z-axis)
///   ω  – argument of periapsis [rad]   (rotation of major axis in orbital plane)
///   ν  – true anomaly          [rad]   (where the satellite is in its orbit NOW)
///
/// ### Singular cases (watch for in interviews)
///
///   e ≈ 0  (circular orbit):    ω undefined → use argument of latitude u = ω+ν
///   i ≈ 0  (equatorial orbit):  Ω undefined → use longitude of periapsis ϖ = Ω+ω
///   e≈0 & i≈0:                  neither Ω nor ω defined → use true longitude
///
/// ### Mean anomaly M and Kepler's equation
///
///   M = n(t − t_peri)   where n = 2π/T = √(μ/a³)  (mean motion)
///   M = E − e·sin(E)    (Kepler's equation; solved iteratively for E)
///   ν = 2·atan2(√(1+e)·sin(E/2), √(1-e)·cos(E/2))
///
/// ### ECI frame
/// X: vernal equinox direction; Z: north celestial pole; Y = Z×X.
//------------------------------------------------------------------------------
#pragma once

#include "StateVector.hpp"
#include "Constants.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace OrbitalMechanics {

/// \brief Keplerian orbital elements.
struct KeplerianElements
{
  double a;   ///< Semi-major axis [m].
  double e;   ///< Eccentricity [-].
  double i;   ///< Inclination [rad].
  double raan;///< Right ascension of ascending node Ω [rad].
  double aop; ///< Argument of periapsis ω [rad].
  double nu;  ///< True anomaly ν [rad].
};

// ── Kepler's equation ────────────────────────────────────────────────────────

/// Solve Kepler's equation  M = E - e·sin(E)  for E, given M and e.
/// Uses Newton-Raphson; converges in ~5 iterations for e < 0.9.
inline double kepler_E_from_M(double M, double e, int max_iter = 50,
                               double tol = 1.0e-12)
{
  // Normalise M to [0, 2π)
  M = std::fmod(M, Constants::TWO_PI);
  if (M < 0.0) M += Constants::TWO_PI;

  double E = (e < 0.8) ? M : Constants::PI; // initial guess

  for (int k = 0; k < max_iter; ++k)
  {
    double f  = E - e * std::sin(E) - M;
    double fp = 1.0 - e * std::cos(E);
    double dE = -f / fp;
    E += dE;
    if (std::abs(dE) < tol) return E;
  }
  throw std::runtime_error("kepler_E_from_M: failed to converge");
}

/// True anomaly ν from eccentric anomaly E.
inline double nu_from_E(double E, double e)
{
  return 2.0 * std::atan2(
    std::sqrt(1.0 + e) * std::sin(E / 2.0),
    std::sqrt(1.0 - e) * std::cos(E / 2.0));
}

/// True anomaly ν from mean anomaly M.
inline double nu_from_M(double M, double e)
{
  return nu_from_E(kepler_E_from_M(M, e), e);
}

// ── Elements → State vector ──────────────────────────────────────────────────

/// Convert Keplerian elements to ECI Cartesian state vector.
///
/// Algorithm:
///   1. Compute p = a(1−e²), r = p/(1+e·cos ν).
///   2. Build position and velocity in the perifocal (PQW) frame.
///   3. Rotate PQW → ECI: R3(−Ω)·R1(−i)·R3(−ω).
///
/// \param el  Keplerian elements.
/// \param mu  Gravitational parameter [m³/s²].
/// \return    Cartesian state [x,y,z,vx,vy,vz] in ECI.
inline StateVector6 elements_to_state(const KeplerianElements& el,
                                      double mu = Constants::MU_EARTH)
{
  const double& a    = el.a;
  const double& e    = el.e;
  const double& inc  = el.i;
  const double& raan = el.raan;
  const double& aop  = el.aop;
  const double& nu   = el.nu;

  const double p    = a * (1.0 - e * e);
  const double r    = p / (1.0 + e * std::cos(nu));
  const double sqmu = std::sqrt(mu / p);

  // Perifocal frame
  const double r_P = r * std::cos(nu);
  const double r_Q = r * std::sin(nu);

  const double v_P = -sqmu * std::sin(nu);
  const double v_Q =  sqmu * (e + std::cos(nu));

  // Rotation matrix elements  R = R3(−Ω)·R1(−i)·R3(−ω)
  const double cO = std::cos(raan), sO = std::sin(raan);
  const double ci = std::cos(inc),  si = std::sin(inc);
  const double co = std::cos(aop),  so = std::sin(aop);

  // Columns of R (perifocal P̂, Q̂ in ECI)
  const double Px =  cO*co - sO*so*ci;
  const double Py =  sO*co + cO*so*ci;
  const double Pz =  so*si;

  const double Qx = -cO*so - sO*co*ci;
  const double Qy = -sO*so + cO*co*ci;
  const double Qz =  co*si;

  return StateVector6{
    Px*r_P + Qx*r_Q,
    Py*r_P + Qy*r_Q,
    Pz*r_P + Qz*r_Q,
    Px*v_P + Qx*v_Q,
    Py*v_P + Qy*v_Q,
    Pz*v_P + Qz*v_Q
  };
}

// ── State vector → Elements ──────────────────────────────────────────────────

/// Convert ECI Cartesian state vector to Keplerian elements.
///
/// Handles prograde, retrograde, and (approximately) circular/equatorial
/// orbits.  Singularity checks: throws if e or sin(i) is too near zero
/// and the degenerate case isn't handled.
///
/// \param s   Cartesian state [x,y,z,vx,vy,vz].
/// \param mu  Gravitational parameter [m³/s²].
/// \return    Keplerian elements.
inline KeplerianElements state_to_elements(const StateVector6& s,
                                           double mu = Constants::MU_EARTH)
{
  const double r_mag = radius(s);
  const double v_mag = speed(s);

  // ── Specific angular momentum h = r × v ──────────────────────────────────
  auto h_vec = angular_momentum(s);
  const double h_mag = norm3(h_vec[0], h_vec[1], h_vec[2]);

  // ── Node vector N = ẑ × h ────────────────────────────────────────────────
  // ẑ = [0,0,1], so N = [−h_y, h_x, 0]
  const double N_x = -h_vec[1];
  const double N_y =  h_vec[0];
  const double N_z =  0.0;
  const double N_mag = std::sqrt(N_x*N_x + N_y*N_y);

  // ── Eccentricity vector e_vec = (v×h)/μ − r̂ ─────────────────────────────
  double vh[3];
  double v3[3] = {s[3], s[4], s[5]};
  cross3(v3, h_vec.data(), vh);  // v × h

  const double ev_x = vh[0]/mu - s[0]/r_mag;
  const double ev_y = vh[1]/mu - s[1]/r_mag;
  const double ev_z = vh[2]/mu - s[2]/r_mag;
  const double e_mag = norm3(ev_x, ev_y, ev_z);

  // ── Specific orbital energy → semi-major axis ─────────────────────────────
  const double eps = 0.5*v_mag*v_mag - mu/r_mag;
  if (eps >= 0.0)
    throw std::domain_error("state_to_elements: non-elliptic orbit (ε≥0)");
  const double a = -mu / (2.0 * eps);

  // ── Inclination ───────────────────────────────────────────────────────────
  const double inc = std::acos(std::clamp(h_vec[2] / h_mag, -1.0, 1.0));

  // ── RAAN ─────────────────────────────────────────────────────────────────
  double raan = 0.0;
  if (N_mag > 1.0e-10)
  {
    raan = std::acos(std::clamp(N_x / N_mag, -1.0, 1.0));
    if (N_y < 0.0) raan = Constants::TWO_PI - raan;
  }

  // ── Argument of periapsis ─────────────────────────────────────────────────
  double aop = 0.0;
  if (N_mag > 1.0e-10 && e_mag > 1.0e-10)
  {
    const double NdotE = N_x*ev_x + N_y*ev_y + N_z*ev_z;
    aop = std::acos(std::clamp(NdotE / (N_mag * e_mag), -1.0, 1.0));
    if (ev_z < 0.0) aop = Constants::TWO_PI - aop;
  }

  // ── True anomaly ──────────────────────────────────────────────────────────
  double nu = 0.0;
  if (e_mag > 1.0e-10)
  {
    const double EdotR = ev_x*s[0] + ev_y*s[1] + ev_z*s[2];
    nu = std::acos(std::clamp(EdotR / (e_mag * r_mag), -1.0, 1.0));
    if (rdotv(s) < 0.0) nu = Constants::TWO_PI - nu; // past apoapsis
  }
  else
  {
    // Circular: use argument of latitude (angle from node to satellite)
    if (N_mag > 1.0e-10)
    {
      const double NdotR = N_x*s[0] + N_y*s[1];
      nu = std::acos(std::clamp(NdotR / (N_mag * r_mag), -1.0, 1.0));
      if (s[2] < 0.0) nu = Constants::TWO_PI - nu;
    }
  }

  return KeplerianElements{a, e_mag, inc, raan, aop, nu};
}

} // namespace OrbitalMechanics
