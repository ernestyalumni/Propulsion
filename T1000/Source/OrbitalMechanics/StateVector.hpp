//------------------------------------------------------------------------------
/// \file StateVector.hpp
/// \brief 6-DOF Cartesian state vector and helper arithmetic.
///
/// The state is [x, y, z, vx, vy, vz] (position then velocity), all in SI
/// units (metres, m/s) in an Earth-Centred Inertial (ECI) frame unless noted.
//------------------------------------------------------------------------------
#pragma once

#include <array>
#include <cmath>
#include <ostream>

namespace OrbitalMechanics {

/// 6-component Cartesian state vector.
using StateVector6 = std::array<double, 6>;

// ── Position / velocity slices ───────────────────────────────────────────────

inline double pos_x(const StateVector6& s) { return s[0]; }
inline double pos_y(const StateVector6& s) { return s[1]; }
inline double pos_z(const StateVector6& s) { return s[2]; }
inline double vel_x(const StateVector6& s) { return s[3]; }
inline double vel_y(const StateVector6& s) { return s[4]; }
inline double vel_z(const StateVector6& s) { return s[5]; }

// ── 3-vector helpers (operate on the first 3 or last 3 components) ───────────

/// Euclidean norm of a 3-vector [a, b, c].
inline double norm3(double a, double b, double c)
{
  return std::sqrt(a*a + b*b + c*c);
}

/// |r| = magnitude of position vector.
inline double radius(const StateVector6& s)
{
  return norm3(s[0], s[1], s[2]);
}

/// |v| = magnitude of velocity vector.
inline double speed(const StateVector6& s)
{
  return norm3(s[3], s[4], s[5]);
}

/// Dot product of two 3-vectors (arrays of length ≥ 3, starting at offset).
inline double dot3(const StateVector6& a, std::size_t oa,
                   const StateVector6& b, std::size_t ob)
{
  return a[oa]*b[ob] + a[oa+1]*b[ob+1] + a[oa+2]*b[ob+2];
}

/// Dot product r·v.
inline double rdotv(const StateVector6& s)
{
  return dot3(s, 0, s, 3);
}

/// Cross product a × b into out (3-component arrays).
inline void cross3(const double* a, const double* b, double* out)
{
  out[0] = a[1]*b[2] - a[2]*b[1];
  out[1] = a[2]*b[0] - a[0]*b[2];
  out[2] = a[0]*b[1] - a[1]*b[0];
}

/// Specific angular momentum vector h = r × v.
inline std::array<double,3> angular_momentum(const StateVector6& s)
{
  std::array<double,3> h;
  cross3(s.data(), s.data()+3, h.data());
  return h;
}

/// |h| = magnitude of specific angular momentum.
inline double angular_momentum_mag(const StateVector6& s)
{
  auto h = angular_momentum(s);
  return norm3(h[0], h[1], h[2]);
}

/// Specific orbital energy ε = v²/2 − μ/r.
inline double specific_energy(const StateVector6& s, double mu)
{
  return 0.5 * speed(s) * speed(s) - mu / radius(s);
}

/// Vis-viva: orbital speed at given r for semi-major axis a.
///   v = sqrt(μ * (2/r − 1/a))
inline double vis_viva(double mu, double r, double a)
{
  return std::sqrt(mu * (2.0/r - 1.0/a));
}

/// Print helper.
inline std::ostream& operator<<(std::ostream& os, const StateVector6& s)
{
  os << "[r=(" << s[0] << ", " << s[1] << ", " << s[2] << ") m  "
     << " v=(" << s[3] << ", " << s[4] << ", " << s[5] << ") m/s]";
  return os;
}

} // namespace OrbitalMechanics
