//------------------------------------------------------------------------------
/// \file   NumerovOrbitDemo.cpp
/// \brief  Astrodynamics demo: Numerov orbit propagation using Cosmos library.
///
/// \details This demo showcases the modular Cosmos architecture:
///   - Algebra::Modules::Vectors::Vector3<double> for 3D vectors
///   - Astrodynamics::TwoBodyAcceleration for Newton + J2 gravity
///   - Astrodynamics::Propagators::NumerovOrbit for PECECE integration
///   - Astrodynamics::SpecificEnergy for orbital energy tracking
///
/// Build:
///   mkdir -p build && cd build && cmake .. && make NumerovOrbitDemo
///
/// Run:
///   ./NumerovOrbitDemo
///
/// Output:
///   numerov_twobody.csv — pure two-body propagation
///   numerov_j2.csv      — two-body + J2 oblateness
//------------------------------------------------------------------------------

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "Algebra/Modules/Vectors/Vector3.h"
#include "Astrodynamics/Propagators/NumerovOrbit.h"
#include "Astrodynamics/TwoBodyAcceleration.h"
#include "Astrodynamics/specific_energy.h"

using Algebra::Modules::Vectors::Vector3;
using Astrodynamics::Propagators::NumerovOrbit;
using Astrodynamics::Propagators::OrbitalState;
using Astrodynamics::TwoBodyAcceleration::AccelerationInputs;
using Astrodynamics::TwoBodyAcceleration::J2Perturbation;
using Astrodynamics::TwoBodyAcceleration::NewtonsGravitation;
using Astrodynamics::TwoBodyAcceleration::TotalAcceleration;
using Astrodynamics::SpecificEnergy;

//------------------------------------------------------------------------------
// Physical constants (SI)
//------------------------------------------------------------------------------
constexpr double MU_EARTH = 3.986004418e14;   // m³/s²
constexpr double R_EARTH  = 6.378137e6;      // m
constexpr double J2         = 1.08263e-3;     // dimensionless

//------------------------------------------------------------------------------
// Analytical J2 nodal regression (rad/s) — Curtis Eq. 4.52
//------------------------------------------------------------------------------
double j2_nodal_regression(double mu, double j2, double r_earth,
                           double a, double e, double i)
{
  const double n = std::sqrt(mu / (a * a * a));
  const double p = a * (1.0 - e * e);
  return n * -1.5 * j2 * (r_earth / p) * (r_earth / p) * std::cos(i);
}

//------------------------------------------------------------------------------
// Analytical J2 perigee advance (rad/s) — Curtis Eq. 4.53
//------------------------------------------------------------------------------
double j2_perigee_advance(double mu, double j2, double r_earth,
                          double a, double e, double i)
{
  const double n = std::sqrt(mu / (a * a * a));
  const double p = a * (1.0 - e * e);
  return n * 0.75 * j2 * (r_earth / p) * (r_earth / p) *
         (5.0 * std::cos(i) * std::cos(i) - 1.0);
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main()
{
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "========================================\n"
            << "Numerov Astrodynamics Demo\n"
            << "Cosmos Library — Modular Orbit Propagation\n"
            << "========================================\n\n";

  //----------------------------------------------------------------------------
  // Initial conditions: 400 km altitude, 28.5° inclination (LEO)
  //----------------------------------------------------------------------------
  const double altitude = 400e3;                          // m
  const double r0       = R_EARTH + altitude;             // m
  const double v_circ   = std::sqrt(MU_EARTH / r0);       // m/s
  const double inclination = 28.5 * M_PI / 180.0;         // rad

  Vector3<double> r0_vec{r0, 0.0, 0.0};
  Vector3<double> v0_vec{0.0,
                         v_circ * std::cos(inclination),
                         v_circ * std::sin(inclination)};

  const double period = 2.0 * M_PI * std::sqrt(r0 * r0 * r0 / MU_EARTH);

  std::cout << "Initial Conditions:\n"
            << "  Altitude: " << altitude / 1000.0 << " km\n"
            << "  Inclination: " << inclination * 180.0 / M_PI << " deg\n"
            << "  v_circ: " << v_circ << " m/s\n"
            << "  Period: " << period / 60.0 << " min\n\n";

  // Analytical J2 rates
  const double omega_dot = j2_nodal_regression(MU_EARTH, J2, R_EARTH,
                                                r0, 0.0, inclination);
  const double argp_dot  = j2_perigee_advance(MU_EARTH, J2, R_EARTH,
                                               r0, 0.0, inclination);

  std::cout << "Analytical J2 (Curtis Eq. 4.52-4.53):\n"
            << "  RAAN drift: " << omega_dot * 180.0 / M_PI * 86400.0
            << " deg/day\n"
            << "  Perigee advance: " << argp_dot * 180.0 / M_PI * 86400.0
            << " deg/day\n\n";

  //----------------------------------------------------------------------------
  // Simulation parameters
  //----------------------------------------------------------------------------
  const double dt = 60.0;           // s (1 minute steps)
  const int n_orbits = 10;
  const int n_steps = static_cast<int>(n_orbits * period / dt);

  std::cout << "Simulation: " << n_orbits << " orbits, dt=" << dt << " s\n\n";

  //----------------------------------------------------------------------------
  // Run both cases: pure two-body, then two-body + J2
  //----------------------------------------------------------------------------
  for (bool use_j2 : {false, true})
  {
    std::cout << "========================================\n"
              << (use_j2 ? "CASE 2: Two-body + J2" : "CASE 1: Pure Two-body")
              << "\n========================================\n";

    // Build acceleration model
    TotalAcceleration<double> accel{};
    accel.add(NewtonsGravitation<double>{MU_EARTH});
    if (use_j2)
    {
      accel.add(J2Perturbation<double>{MU_EARTH, J2, R_EARTH});
    }

    // Initial state
    OrbitalState<double> s0{0.0, r0_vec, v0_vec};

    // Create propagator (startup happens in constructor via StormerStep)
    NumerovOrbit<double> prop{std::move(accel), s0, dt};

    // Initial energy
    SpecificEnergy<double> energy_calc{MU_EARTH};
    const double E0 = energy_calc(s0.r, s0.v);
    std::cout << "Initial energy: " << std::scientific << E0 << " J/kg\n";

    // Open output file
    const std::string filename = use_j2 ? "numerov_j2.csv" : "numerov_twobody.csv";
    std::ofstream out(filename);
    out << std::scientific << std::setprecision(15);
    out << "t,orbit,r_x,r_y,r_z,energy,error\n";

    // Write initial state (t=0)
    out << s0.t << ",0," << s0.r.x() << "," << s0.r.y() << ","
        << s0.r.z() << "," << E0 << ",0\n";

    // Propagation loop
    double max_err = 0.0;
    int last_orbit = 0;

    for (int i = 2; i <= n_steps; ++i)
    {
      prop.step();
      const auto s = prop.get_state();

      const double E = energy_calc(s.r, s.v);
      const double err = std::abs((E - E0) / E0);
      if (err > max_err) max_err = err;

      const int orbit = static_cast<int>(s.t / period);

      // Write every 10th step
      if (i % 10 == 0)
      {
        out << s.t << "," << orbit << ","
            << s.r.x() << "," << s.r.y() << "," << s.r.z() << ","
            << E << "," << err << "\n";
      }

      // Log orbit transitions
      if (orbit != last_orbit)
      {
        std::cout << "  Orbit " << orbit
                  << ": r=" << std::fixed << std::setprecision(2)
                  << s.r.norm() / 1000.0 << " km"
                  << ", err=" << std::scientific << err << "\n";
        last_orbit = orbit;
      }
    }

    out.close();

    std::cout << "\nResults:\n"
              << "  Max energy error: " << std::scientific << max_err << "\n"
              << "  Output: " << filename << "\n\n";
  }

  std::cout << "========================================\n"
            << "Demo Complete!\n"
            << "========================================\n";

  return 0;
}