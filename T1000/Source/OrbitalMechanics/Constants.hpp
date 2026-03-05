//------------------------------------------------------------------------------
/// \file Constants.hpp
/// \brief Physical and astronomical constants for orbital mechanics.
//------------------------------------------------------------------------------
#pragma once

namespace OrbitalMechanics {
namespace Constants {

/// Earth standard gravitational parameter μ = GM  [m³/s²]
/// Source: WGS84 / EGM2008
constexpr double MU_EARTH = 3.986004418e14;

/// Earth equatorial radius  [m]
constexpr double R_EARTH = 6.3781e6;

/// Earth's rotation rate  [rad/s]
constexpr double OMEGA_EARTH = 7.2921150e-5;

/// Sun standard gravitational parameter  [m³/s²]
constexpr double MU_SUN = 1.32712440018e20;

/// Moon standard gravitational parameter  [m³/s²]
constexpr double MU_MOON = 4.9048695e12;

/// AU in metres
constexpr double AU = 1.495978707e11;

/// π
constexpr double PI = 3.141592653589793238462643383279502884;

/// 2π
constexpr double TWO_PI = 2.0 * PI;

/// Degrees → radians
constexpr double DEG2RAD = PI / 180.0;

/// Radians → degrees
constexpr double RAD2DEG = 180.0 / PI;

} // namespace Constants
} // namespace OrbitalMechanics
