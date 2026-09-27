//------------------------------------------------------------------------------
/// \file Scenarios.h
/// \brief Named scenarios shared by the unit tests and the golden-vector
///   emitter. The Rust twin (cosmos_propulsion::solid_rocket_motor::scenarios)
///   defines the same scenarios with the same arithmetic; change both together.
///
/// Every number is either quoted from a book (with the page) or labeled
/// "assumed" / "illustrative". Illustrative values are not data for a named
/// propellant.
//------------------------------------------------------------------------------
#ifndef PROPULSION_SOLID_ROCKET_MOTOR_SCENARIOS_H
#define PROPULSION_SOLID_ROCKET_MOTOR_SCENARIOS_H

#include "Propulsion/SolidRocketMotor/BurningRate.h"
#include "Propulsion/SolidRocketMotor/CombustionProducts.h"
#include "Propulsion/SolidRocketMotor/GasGeneratorValveSystem.h"
#include "Propulsion/SolidRocketMotor/Grain.h"
#include "Propulsion/SolidRocketMotor/GrainChamber.h"
#include "Propulsion/SolidRocketMotor/PintleValve.h"
#include "Propulsion/SolidRocketMotor/SolidRocketMotor.h"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>
#include <vector>

namespace Propulsion
{
namespace SolidRocketMotor
{
namespace Scenarios
{

inline constexpr double standard_atmosphere_pressure {101325.0};

//------------------------------------------------------------------------------
/// Huzel & Huang (1992), p. 116: solid-propellant start cartridge. Quoted:
/// 1000 psia chamber pressure, 4.7 lb/s, 2550 F, c* = 4260 ft/s, ~1.0 s.
//------------------------------------------------------------------------------
struct HuzelCartridge
{
  static constexpr double pascals_per_psi {6894.757293168};
  static constexpr double kilograms_per_pound {0.45359237};
  static constexpr double meters_per_foot {0.3048};

  static constexpr double chamber_pressure {1000.0 * pascals_per_psi};
  static constexpr double mass_flow {4.7 * kilograms_per_pound};
  static constexpr double characteristic_velocity {4260.0 * meters_per_foot};
  // 2550 F -> K.
  static constexpr double stagnation_temperature {
    (2550.0 - 32.0) * 5.0 / 9.0 + 273.15};
  static constexpr double burn_time {1.0};

  // Assumed (not given on p. 116):
  static constexpr double heat_capacity_ratio {1.25};
  static constexpr double propellant_density {1600.0};
  static constexpr double burning_rate_at_design {0.02};
  static constexpr double pressure_exponent {0.4};
  static constexpr double sigma_p {0.0018};
  static constexpr double reference_temperature {294.15};
  static constexpr double free_volume {0.0015};

  static CombustionProducts<double> products()
  {
    return CombustionProducts<double>::from_characteristic_velocity(
      characteristic_velocity,
      stagnation_temperature,
      heat_capacity_ratio);
  }

  /// A_t = m_dot c* / p_1 (Sutton Eq. 3-32).
  static double throat_area()
  {
    return mass_flow * characteristic_velocity / chamber_pressure;
  }

  /// A_b = m_dot / (rho_b r) (Sutton Eq. 12-1), end-burning face radius.
  static double grain_radius()
  {
    const double burning_area {
      mass_flow / (propellant_density * burning_rate_at_design)};
    return std::sqrt(burning_area / std::numbers::pi);
  }

  /// a = r / p^n at the design point.
  static double reference_coefficient()
  {
    return burning_rate_at_design / std::pow(chamber_pressure, pressure_exponent);
  }

  static GrainChamber<double, EndBurningGrain<double>> chamber()
  {
    return GrainChamber<double, EndBurningGrain<double>>{
      EndBurningGrain<double>{
        grain_radius(),
        burning_rate_at_design * burn_time,
        free_volume},
      propellant_density,
      SaintRobertBurningRate<double>{
        reference_coefficient(),
        pressure_exponent,
        TemperatureSensitivity<double>::exponential(
          sigma_p,
          reference_temperature)},
      std::nullopt,
      products(),
      reference_temperature};
  }

  /// The cartridge alone, exhausting through its outlet orifice (epsilon = 1).
  static SolidRocketMotor<double, EndBurningGrain<double>> motor()
  {
    return SolidRocketMotor<double, EndBurningGrain<double>>{
      chamber(),
      throat_area(),
      throat_area(),
      1.0,
      standard_atmosphere_pressure};
  }

  static constexpr double initial_pressure {1.0e6};
  static constexpr double step {5.0e-5};
  static constexpr std::size_t step_count {26000};
  static constexpr std::size_t sample_interval {500};
};

//------------------------------------------------------------------------------
/// Illustrative tubular booster segment with erosive burning and the Williams
/// explosion-temperature law. Composite-like products.
//------------------------------------------------------------------------------
struct TubularBooster
{
  static constexpr double inner_radius {0.05};
  static constexpr double outer_radius {0.10};
  static constexpr double length {1.0};
  static constexpr double free_volume {0.003};
  static constexpr double propellant_density {1760.0};
  static constexpr double reference_pressure {7.0e6};
  static constexpr double burning_rate_at_reference {0.007};
  static constexpr double pressure_exponent {0.35};
  static constexpr double reference_temperature {294.15};
  // Williams 7-41 with sigma_p(T_ref) = 1 / (T_e - T_ref) = 0.002 / K.
  static constexpr double explosion_temperature {794.15};
  static constexpr double grain_temperature {283.15};
  static constexpr double erosive_alpha {2.0e-5};
  // Sutton p. 454: beta "about 53" in SI.
  static constexpr double erosive_beta {53.0};
  static constexpr double stagnation_temperature {3300.0};
  static constexpr double molar_mass {0.029};
  static constexpr double heat_capacity_ratio {1.18};
  static constexpr double throat_area {1.05e-3};
  static constexpr double exit_area {8.4e-3};
  static constexpr double nozzle_discharge_coefficient {0.98};

  static double reference_coefficient()
  {
    return burning_rate_at_reference /
      std::pow(reference_pressure, pressure_exponent);
  }

  static GrainChamber<double, TubularGrain<double>> chamber()
  {
    return GrainChamber<double, TubularGrain<double>>{
      TubularGrain<double>{
        inner_radius,
        outer_radius,
        length,
        true,
        free_volume},
      propellant_density,
      SaintRobertBurningRate<double>{
        reference_coefficient(),
        pressure_exponent,
        TemperatureSensitivity<double>::explosion_temperature(
          explosion_temperature,
          reference_temperature)},
      ErosiveBurning<double>{erosive_alpha, erosive_beta},
      CombustionProducts<double>{
        stagnation_temperature,
        molar_mass,
        heat_capacity_ratio},
      grain_temperature};
  }

  static SolidRocketMotor<double, TubularGrain<double>> motor()
  {
    return SolidRocketMotor<double, TubularGrain<double>>{
      chamber(),
      throat_area,
      exit_area,
      nozzle_discharge_coefficient,
      standard_atmosphere_pressure};
  }

  static constexpr double initial_pressure {1.0e6};
  static constexpr double step {2.0e-4};
  static constexpr std::size_t step_count {45000};
  static constexpr std::size_t sample_interval {1000};
};

//------------------------------------------------------------------------------
/// The Huzel cartridge as a gas generator feeding a plenum through its outlet
/// orifice, throttled by four pintle valves. Valves 0 and 1 stay open; valves 2
/// and 3 close between 0.40 s and 0.45 s. With all four open the orifice is
/// choked; with two, the plenum pressure exceeds the critical ratio, the
/// orifice unchokes, and the grain-chamber pressure rises (Sutton Fig. 12-27).
//------------------------------------------------------------------------------
struct CartridgeWithFourPintleValves
{
  // Assumed plenum and valves:
  static constexpr double plenum_volume {0.004};
  static constexpr double wall_heat_conductance {0.0};
  static constexpr double wall_temperature {300.0};
  static constexpr double valve_throat_radius {0.010};
  static constexpr double valve_expansion_ratio_full_open {4.0};
  static constexpr double valve_discharge_coefficient {0.95};
  static constexpr double close_start {0.40};
  static constexpr double close_end {0.45};
  static constexpr std::size_t valve_count {4};

  static std::vector<PintleValve<double>> valves()
  {
    const double full_open_area {
      std::numbers::pi * valve_throat_radius * valve_throat_radius};
    std::vector<PintleValve<double>> result {};
    for (std::size_t i {0}; i < valve_count; ++i)
    {
      const OpeningSchedule<double> schedule {i < 2 ?
        OpeningSchedule<double>::constant(1.0) :
        OpeningSchedule<double>{{{close_start, 1.0}, {close_end, 0.0}}}};
      result.push_back(PintleValve<double>{
        valve_throat_radius,
        valve_expansion_ratio_full_open * full_open_area,
        valve_discharge_coefficient,
        schedule});
    }
    return result;
  }

  static GasGeneratorValveSystem<double, EndBurningGrain<double>> system()
  {
    return GasGeneratorValveSystem<double, EndBurningGrain<double>>{
      HuzelCartridge::chamber(),
      HuzelCartridge::throat_area(),
      1.0,
      Plenum<double>{plenum_volume, wall_heat_conductance, wall_temperature},
      valves(),
      standard_atmosphere_pressure};
  }

  static constexpr double initial_chamber_pressure {1.0e6};
  static constexpr double initial_plenum_pressure {standard_atmosphere_pressure};
  static constexpr double initial_plenum_temperature {wall_temperature};
  static constexpr double step {5.0e-5};
  static constexpr std::size_t step_count {26000};
  static constexpr std::size_t sample_interval {500};
};

} // namespace Scenarios
} // namespace SolidRocketMotor
} // namespace Propulsion

#endif // PROPULSION_SOLID_ROCKET_MOTOR_SCENARIOS_H
