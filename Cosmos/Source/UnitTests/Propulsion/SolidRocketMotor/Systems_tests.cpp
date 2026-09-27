#include "Propulsion/SolidRocketMotor/Scenarios.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>
#include <vector>

using Propulsion::SolidRocketMotor::CombustionProducts;
using Propulsion::SolidRocketMotor::critical_pressure_ratio;
using Propulsion::SolidRocketMotor::EndBurningGrain;
using Propulsion::SolidRocketMotor::GasGeneratorValveSystem;
using Propulsion::SolidRocketMotor::GrainChamber;
using Propulsion::SolidRocketMotor::OpeningSchedule;
using Propulsion::SolidRocketMotor::PintleValve;
using Propulsion::SolidRocketMotor::Plenum;
using Propulsion::SolidRocketMotor::TemperatureSensitivity;
using Propulsion::SolidRocketMotor::SaintRobertBurningRate;
using Propulsion::SolidRocketMotor::Scenarios::CartridgeWithFourPintleValves;
using Propulsion::SolidRocketMotor::Scenarios::HuzelCartridge;
using Propulsion::SolidRocketMotor::Scenarios::TubularBooster;

namespace GoogleUnitTests
{
namespace Propulsion
{
namespace SolidRocketMotor
{

template <typename Sample>
const Sample& sample_nearest(const std::vector<Sample>& samples, const double t)
{
  std::size_t best {0};
  for (std::size_t i {1}; i < samples.size(); ++i)
  {
    if (std::abs(samples[i].time - t) < std::abs(samples[best].time - t))
    {
      best = i;
    }
  }
  return samples[best];
}

//------------------------------------------------------------------------------
/// Huzel & Huang p. 116: the cartridge sized from the quoted numbers settles
/// at the pressure where (rho_b - rho) A_b a p^n = A_t p / c* (Hill & Peterson
/// Eq. 12.28 with dp/dt = 0), within 1.5% of 1000 psia and 4.7 lb/s. The gap
/// is the (rho_b - rho) filling term, which the quoted sizing ignores.
//------------------------------------------------------------------------------
TEST(SolidRocketMotorTests, HuzelCartridgeReachesItsEquilibrium)
{
  const auto motor = HuzelCartridge::motor();
  const auto samples = motor.simulate(
    HuzelCartridge::initial_pressure,
    HuzelCartridge::step,
    HuzelCartridge::step_count,
    HuzelCartridge::sample_interval);
  const auto& steady = sample_nearest(samples, 0.5);

  const auto& chamber = motor.chamber();
  const auto& gas = chamber.products();
  const double p {steady.chamber_pressure};
  const double generation {
    (chamber.propellant_density() - gas.density_at(p)) *
      chamber.grain().burning_area(steady.burned_web) *
      chamber.burning_rate(p, steady.burned_web)};
  const double outflow {HuzelCartridge::throat_area() * p /
    gas.characteristic_velocity()};
  EXPECT_NEAR(generation, outflow, 1.0e-9 * outflow);

  EXPECT_NEAR(p, HuzelCartridge::chamber_pressure,
    0.015 * HuzelCartridge::chamber_pressure);
  EXPECT_NEAR(steady.nozzle_mass_flow, HuzelCartridge::mass_flow,
    0.015 * HuzelCartridge::mass_flow);

  // Burnout close to the quoted ~1.0 s, then blow-down to ambient.
  EXPECT_LT(sample_nearest(samples, 1.2).chamber_pressure, 1.1e5);
}

//------------------------------------------------------------------------------
/// solid-ballistics.tex Prop. sb-stability: a perturbation decays at
/// 1 / tau, tau = V c* / ((1 - n) R T_0 A_t). The (rho_b - rho) term and the
/// slowly growing V shift the measured rate by about a percent.
//------------------------------------------------------------------------------
TEST(SolidRocketMotorTests, PressurePerturbationDecaysAtTheStabilityRate)
{
  const auto motor = HuzelCartridge::motor();
  const auto settled = motor.simulate(
    HuzelCartridge::initial_pressure, HuzelCartridge::step, 4000, 4000);
  const double p_eq {settled.back().chamber_pressure};

  const auto samples = motor.simulate(1.05 * p_eq, 1.0e-5, 3000, 1);
  const auto& gas = motor.chamber().products();
  const double tau {
    motor.chamber().grain().gas_volume(0.0) * gas.characteristic_velocity() /
    ((1.0 - HuzelCartridge::pressure_exponent) * gas.specific_gas_constant() *
      gas.stagnation_temperature() * HuzelCartridge::throat_area())};

  const double t1 {0.25 * tau};
  const double t2 {1.25 * tau};
  const double d1 {sample_nearest(samples, t1).chamber_pressure - p_eq};
  const double d2 {sample_nearest(samples, t2).chamber_pressure - p_eq};
  const double measured {
    (sample_nearest(samples, t2).time - sample_nearest(samples, t1).time) /
    std::log(d1 / d2)};
  EXPECT_NEAR(measured, tau, 0.05 * tau);
}

//------------------------------------------------------------------------------
/// RK4 on the ignition transient converges at fourth order under step
/// halving: the error ratio between successive halvings is about 2^4.
//------------------------------------------------------------------------------
TEST(SolidRocketMotorTests, IgnitionTransientConvergesAtFourthOrder)
{
  const auto motor = HuzelCartridge::motor();
  const double end_time {0.02};
  const auto pressure_at_end = [&](const std::size_t steps)
  {
    return motor.simulate(
      HuzelCartridge::initial_pressure,
      end_time / static_cast<double>(steps),
      steps,
      steps).back().chamber_pressure;
  };
  const double coarse {pressure_at_end(10)};
  const double medium {pressure_at_end(20)};
  const double fine {pressure_at_end(40)};
  const double observed_order {
    std::log2(std::abs(coarse - medium) / std::abs(medium - fine))};
  EXPECT_NEAR(observed_order, 4.0, 0.3);
}

//------------------------------------------------------------------------------
/// Sutton Eq. 12-14: pi_K = sigma_p / (1 - n).
//------------------------------------------------------------------------------
TEST(SolidRocketMotorTests, TemperatureSensitivityOfPressureIsSigmaOverOneMinusN)
{
  const auto run = [](const double grain_temperature)
  {
    const auto base = HuzelCartridge::chamber();
    const GrainChamber<double, EndBurningGrain<double>> chamber {
      base.grain(),
      base.propellant_density(),
      base.burning_rate_law(),
      std::nullopt,
      base.products(),
      grain_temperature};
    const ::Propulsion::SolidRocketMotor::SolidRocketMotor<
      double, EndBurningGrain<double>> motor {
      chamber,
      HuzelCartridge::throat_area(),
      HuzelCartridge::throat_area(),
      1.0,
      101325.0};
    return motor.simulate(1.0e6, 5.0e-5, 6000, 6000).back().chamber_pressure;
  };
  const double delta {10.0};
  const double t_ref {HuzelCartridge::reference_temperature};
  const double pi_k {
    (std::log(run(t_ref + delta)) - std::log(run(t_ref - delta))) /
      (2.0 * delta)};
  const double expected {
    HuzelCartridge::sigma_p / (1.0 - HuzelCartridge::pressure_exponent)};
  EXPECT_NEAR(pi_k, expected, 0.02 * expected);
}

//------------------------------------------------------------------------------
/// Tubular booster: mass is conserved to round-off, the grain burns out, the
/// chamber blows down, erosive burning raises the early burning rate, and the
/// progressive grain gives a rising pressure trace.
//------------------------------------------------------------------------------
TEST(SolidRocketMotorTests, TubularBoosterConservesMassAndBurnsOut)
{
  const auto motor = TubularBooster::motor();
  const auto samples = motor.simulate(
    TubularBooster::initial_pressure,
    TubularBooster::step,
    TubularBooster::step_count,
    TubularBooster::sample_interval);

  for (const auto& s : samples)
  {
    EXPECT_LT(std::abs(s.mass_residual), 1.0e-9 * (1.0 + s.burned_propellant_mass));
  }
  const auto& last = samples.back();
  EXPECT_DOUBLE_EQ(last.burned_web, motor.chamber().grain().web());
  EXPECT_LT(last.chamber_pressure, 1.1e5);

  const double loaded {
    TubularBooster::propellant_density *
      motor.chamber().grain().propellant_volume(0.0)};
  EXPECT_NEAR(last.burned_propellant_mass, loaded, 1.0e-9 * loaded);

  EXPECT_LT(
    sample_nearest(samples, 1.0).chamber_pressure,
    sample_nearest(samples, 5.0).chamber_pressure);

  const auto& early = sample_nearest(samples, 1.0);
  const double base_rate {motor.chamber().burning_rate_law().rate(
    early.chamber_pressure, TubularBooster::grain_temperature)};
  EXPECT_GT(early.burning_rate, 1.05 * base_rate);
}

//------------------------------------------------------------------------------
/// Gas generator with four pintle valves (Sutton Fig. 12-27 behavior).
//------------------------------------------------------------------------------
TEST(GasGeneratorValveSystemTests, ClosingValvesRaisesPlenumAndThenChamberPressure)
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

  for (const auto& s : samples)
  {
    EXPECT_LT(std::abs(s.mass_residual), 1.0e-9 * (1.0 + s.burned_propellant_mass));
  }

  const auto& gas = system.chamber().products();
  const double gamma {gas.heat_capacity_ratio()};
  const auto& open = sample_nearest(samples, 0.35);
  const auto& throttled = sample_nearest(samples, 0.90);

  // Adiabatic plenum fed by one gas: T -> T_0.
  EXPECT_NEAR(open.plenum_temperature, gas.stagnation_temperature(),
    1.0e-6 * gas.stagnation_temperature());
  // Steady state: generation = orifice flow = valve flow.
  EXPECT_NEAR(open.orifice_mass_flow, open.valve_mass_flow,
    1.0e-6 * open.valve_mass_flow);
  // All valves open: orifice choked.
  EXPECT_LT(open.plenum_pressure / open.chamber_pressure,
    critical_pressure_ratio(gamma));
  // p_plenum = m_dot sqrt(R T_0) / (Gamma sum C_d A).
  const double open_area {
    C::valve_count * C::valve_discharge_coefficient *
      system.valves()[0].full_open_area()};
  EXPECT_NEAR(
    open.plenum_pressure,
    open.valve_mass_flow * gas.characteristic_velocity() / open_area,
    1.0e-6 * open.plenum_pressure);

  // Two valves closed: plenum pressure up, orifice unchoked, chamber up.
  EXPECT_GT(throttled.plenum_pressure, 1.8 * open.plenum_pressure);
  EXPECT_GT(throttled.plenum_pressure / throttled.chamber_pressure,
    critical_pressure_ratio(gamma));
  EXPECT_GT(throttled.chamber_pressure, 1.02 * open.chamber_pressure);
  EXPECT_NEAR(throttled.orifice_mass_flow, throttled.valve_mass_flow,
    1.0e-5 * throttled.valve_mass_flow);
}

//------------------------------------------------------------------------------
/// N is a runtime size: the same gas generator with 1, 2, 3 and 7 identical
/// open valves of fixed total area settles to the same plenum pressure, and
/// with no valve open the plenum only fills.
//------------------------------------------------------------------------------
TEST(GasGeneratorValveSystemTests, ValveCountIsVariable)
{
  const double total_area {1.2e-3};
  std::vector<double> plenum_pressures {};
  for (const std::size_t count : {1u, 2u, 3u, 7u})
  {
    const double radius {std::sqrt(total_area / (count * std::numbers::pi))};
    std::vector<PintleValve<double>> valves {};
    for (std::size_t i {0}; i < count; ++i)
    {
      valves.push_back(PintleValve<double>{
        radius,
        4.0 * total_area / count,
        1.0,
        OpeningSchedule<double>::constant(1.0)});
    }
    const GasGeneratorValveSystem<double, EndBurningGrain<double>> system {
      HuzelCartridge::chamber(),
      HuzelCartridge::throat_area(),
      1.0,
      Plenum<double>{0.004, 0.0, 300.0},
      valves,
      101325.0};
    const auto samples = system.simulate(
      system.initial_state(1.0e6, 101325.0, 300.0), 5.0e-5, 8000, 8000);
    plenum_pressures.push_back(samples.back().plenum_pressure);
  }
  for (const double p : plenum_pressures)
  {
    EXPECT_NEAR(p, plenum_pressures.front(), 1.0e-9 * p);
  }

  const GasGeneratorValveSystem<double, EndBurningGrain<double>> sealed {
    HuzelCartridge::chamber(),
    HuzelCartridge::throat_area(),
    1.0,
    Plenum<double>{0.004, 0.0, 300.0},
    {PintleValve<double>{0.01, 4.0e-4, 1.0,
      OpeningSchedule<double>::constant(0.0)}},
    101325.0};
  const auto filling = sealed.simulate(
    sealed.initial_state(1.0e6, 101325.0, 300.0), 5.0e-5, 400, 100);
  for (std::size_t i {1}; i < filling.size(); ++i)
  {
    EXPECT_GT(filling[i].plenum_pressure, filling[i - 1].plenum_pressure);
    EXPECT_DOUBLE_EQ(filling[i].expelled_mass, 0.0);
  }
}

//------------------------------------------------------------------------------
/// Wall heat loss lowers the plenum temperature below T_0.
//------------------------------------------------------------------------------
TEST(GasGeneratorValveSystemTests, WallHeatLossCoolsThePlenum)
{
  using C = CartridgeWithFourPintleValves;
  const GasGeneratorValveSystem<double, EndBurningGrain<double>> cooled {
    HuzelCartridge::chamber(),
    HuzelCartridge::throat_area(),
    1.0,
    Plenum<double>{C::plenum_volume, 500.0, C::wall_temperature},
    C::valves(),
    101325.0};
  const auto samples = cooled.simulate(
    cooled.initial_state(1.0e6, 101325.0, 300.0), 5.0e-5, 6000, 6000);
  EXPECT_LT(samples.back().plenum_temperature,
    0.99 * HuzelCartridge::stagnation_temperature);
  EXPECT_LT(std::abs(samples.back().mass_residual), 1.0e-9);
}

} // namespace SolidRocketMotor
} // namespace Propulsion
} // namespace GoogleUnitTests
