//------------------------------------------------------------------------------
/// \file emit_solid_rocket_motor_golden.cpp
/// \brief Emit golden vectors for the solid rocket motor / gas generator from
///   the C++ reference (Cosmos/Source/Propulsion/SolidRocketMotor), for the
///   Rust twin (cosmos_propulsion::solid_rocket_motor) to check against.
///
/// Build and run from the repository root:
///   g++ -std=c++20 -O2 -I Cosmos/Source
///     Cosmos/Rust/tools/emit_solid_rocket_motor_golden.cpp -o /tmp/emit_srm  (one line)
///   for table in components huzel_cartridge tubular_booster gas_generator; do
///     /tmp/emit_srm $table > Cosmos/Rust/golden/solid_rocket_motor_$table.tsv
///   done
///
/// Never regenerate these to make a failing Rust comparison pass (story 16).
//------------------------------------------------------------------------------
#include "Propulsion/SolidRocketMotor/Scenarios.h"

#include <array>
#include <cstdio>
#include <string>
#include <vector>

using namespace Propulsion::SolidRocketMotor;
using namespace Propulsion::SolidRocketMotor::Scenarios;

namespace
{

constexpr int input_columns {8};

void component_row(const char* kind, std::vector<double> inputs, const double value)
{
  inputs.resize(input_columns, 0.0);
  std::printf("%s", kind);
  for (const double x : inputs)
  {
    std::printf("\t%.17g", x);
  }
  std::printf("\t%.17g\n", value);
}

void emit_components()
{
  std::printf("# source: Cosmos/Source/Propulsion/SolidRocketMotor (components)\n");
  std::printf("kind\tx0\tx1\tx2\tx3\tx4\tx5\tx6\tx7\tvalue\n");

  // restriction: gamma, molar mass, area, p_a, T_a, p_b, T_b.
  for (const double gamma : {1.15, 1.25, 1.4})
  {
    const CombustionProducts<double> gas {2000.0, 0.022, gamma};
    for (const double ratio : {0.0, 0.1, 0.5, 0.55, 0.6, 0.9, 0.999, 1.0, 1.3, 4.0})
    {
      const double p_a {5.0e6};
      const double p_b {ratio * p_a};
      component_row("restriction", {gamma, 0.022, 3.0e-4, p_a, 1800.0, p_b, 900.0},
        restriction_mass_flow(gas, 3.0e-4, p_a, 1800.0, p_b, 900.0));
    }
  }

  // mach: gamma, epsilon.
  for (const double gamma : {1.15, 1.25, 1.4})
  {
    for (const double epsilon : {1.0, 1.01, 2.0, 8.0, 40.0, 150.0})
    {
      component_row("mach", {gamma, epsilon},
        supersonic_exit_mach(epsilon, gamma));
    }
  }

  // thrust: gamma, molar mass, T_c, C_d, A_t, A_e, p_c, p_a.
  for (const double gamma : {1.18, 1.25})
  {
    const CombustionProducts<double> gas {3000.0, 0.025, gamma};
    for (const double p_c : {5.0e4, 1.5e5, 2.0e5, 1.0e6, 7.0e6})
    {
      for (const double p_a : {0.0, 101325.0})
      {
        for (const double epsilon : {1.0, 6.0})
        {
          component_row("thrust",
            {gamma, 0.025, 2900.0, 0.97, 1.0e-3, epsilon * 1.0e-3, p_c, p_a},
            ideal_thrust(gas, 0.97, 1.0e-3, epsilon * 1.0e-3, p_c, 2900.0, p_a));
        }
      }
    }
  }

  // erosive: r_0, alpha, beta, rho_b, A_b, A_p, D.
  for (const double r0 : {0.0, 1.0e-3, 7.0e-3, 2.0e-2})
  {
    for (const double alpha : {0.0, 2.0e-5, 1.0e-4})
    {
      for (const double port_area : {2.0e-3, 7.85e-3, 3.0e-2})
      {
        const ErosiveBurning<double> erosive {alpha, 53.0};
        component_row("erosive", {r0, alpha, 53.0, 1760.0, 0.314, port_area, 0.1},
          erosive_burning_rate(r0, erosive, 1760.0, 0.314, port_area, 0.1));
      }
    }
  }

  // temperature: law (0 exponential, 1 explosion temperature), parameter,
  // T_ref, T_b.
  for (const double t_b : {233.15, 294.15, 344.15})
  {
    component_row("temperature", {0.0, 0.0018, 294.15, t_b},
      TemperatureSensitivity<double>::exponential(0.0018, 294.15)
        .coefficient_factor(t_b));
    component_row("temperature", {1.0, 794.15, 294.15, t_b},
      TemperatureSensitivity<double>::explosion_temperature(794.15, 294.15)
        .coefficient_factor(t_b));
  }

  // tubular_area / tubular_volume: a, b, L, ends inhibited, free volume, y.
  for (const double inhibited : {1.0, 0.0})
  {
    const TubularGrain<double> grain {0.05, 0.10, 1.0, inhibited > 0.5, 0.003};
    for (const double y : {0.0, 0.01, 0.049999, 0.05, 0.06})
    {
      component_row("tubular_area", {0.05, 0.10, 1.0, inhibited, 0.003, y},
        grain.burning_area(y));
      component_row("tubular_volume", {0.05, 0.10, 1.0, inhibited, 0.003, y},
        grain.gas_volume(y));
    }
  }

  // pintle_area: throat radius, opening.
  for (const double x : {0.0, 0.1, 0.5, 0.9, 1.0})
  {
    const PintleValve<double> valve {0.01, 4.0e-4, 0.95,
      OpeningSchedule<double>::constant(1.0)};
    component_row("pintle_area", {0.01, x}, valve.flow_area(x));
  }
}

void emit_motor(const char* name, const std::vector<MotorSample<double>>& samples)
{
  std::printf("# source: Cosmos/Source/Propulsion/SolidRocketMotor/Scenarios.h %s\n", name);
  std::printf(
    "time\tburned_web\tchamber_pressure\tburning_rate\tgeneration_rate\t"
    "nozzle_mass_flow\tthrust\texpelled_mass\ttotal_impulse\t"
    "burned_propellant_mass\tmass_residual\n");
  for (const auto& s : samples)
  {
    const std::array<double, 11> row {
      s.time, s.burned_web, s.chamber_pressure, s.burning_rate,
      s.generation_rate, s.nozzle_mass_flow, s.thrust, s.expelled_mass,
      s.total_impulse, s.burned_propellant_mass, s.mass_residual};
    for (std::size_t i {0}; i < row.size(); ++i)
    {
      std::printf(i == 0 ? "%.17g" : "\t%.17g", row[i]);
    }
    std::printf("\n");
  }
}

void emit_gas_generator()
{
  using C = CartridgeWithFourPintleValves;
  const auto system = C::system();
  const auto samples = system.simulate(
    system.initial_state(
      C::initial_chamber_pressure,
      C::initial_plenum_pressure,
      C::initial_plenum_temperature),
    C::step,
    C::step_count,
    C::sample_interval);
  std::printf("# source: Cosmos/Source/Propulsion/SolidRocketMotor/Scenarios.h "
    "CartridgeWithFourPintleValves\n");
  std::printf(
    "time\tburned_web\tchamber_pressure\tplenum_pressure\tplenum_temperature\t"
    "burning_rate\tgeneration_rate\torifice_mass_flow\tvalve_mass_flow\t"
    "thrust\texpelled_mass\ttotal_impulse\tburned_propellant_mass\t"
    "mass_residual\n");
  for (const auto& s : samples)
  {
    const std::array<double, 14> row {
      s.time, s.burned_web, s.chamber_pressure, s.plenum_pressure,
      s.plenum_temperature, s.burning_rate, s.generation_rate,
      s.orifice_mass_flow, s.valve_mass_flow, s.thrust, s.expelled_mass,
      s.total_impulse, s.burned_propellant_mass, s.mass_residual};
    for (std::size_t i {0}; i < row.size(); ++i)
    {
      std::printf(i == 0 ? "%.17g" : "\t%.17g", row[i]);
    }
    std::printf("\n");
  }
}

} // namespace

int main(int argc, char** argv)
{
  const std::string table {argc > 1 ? argv[1] : ""};
  if (table == "components")
  {
    emit_components();
  }
  else if (table == "huzel_cartridge")
  {
    emit_motor("HuzelCartridge", HuzelCartridge::motor().simulate(
      HuzelCartridge::initial_pressure,
      HuzelCartridge::step,
      HuzelCartridge::step_count,
      HuzelCartridge::sample_interval));
  }
  else if (table == "tubular_booster")
  {
    emit_motor("TubularBooster", TubularBooster::motor().simulate(
      TubularBooster::initial_pressure,
      TubularBooster::step,
      TubularBooster::step_count,
      TubularBooster::sample_interval));
  }
  else if (table == "gas_generator")
  {
    emit_gas_generator();
  }
  else
  {
    std::fprintf(stderr,
      "usage: %s components|huzel_cartridge|tubular_booster|gas_generator\n",
      argv[0]);
    return 1;
  }
  return 0;
}
