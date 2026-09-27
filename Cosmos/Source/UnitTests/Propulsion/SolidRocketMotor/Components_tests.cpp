#include "Propulsion/SolidRocketMotor/BurningRate.h"
#include "Propulsion/SolidRocketMotor/CombustionProducts.h"
#include "Propulsion/SolidRocketMotor/CompressibleFlow.h"
#include "Propulsion/SolidRocketMotor/Grain.h"
#include "Propulsion/SolidRocketMotor/PintleValve.h"

#include "gtest/gtest.h"

#include <cmath>
#include <numbers>

using Propulsion::SolidRocketMotor::area_ratio_at_mach;
using Propulsion::SolidRocketMotor::CombustionProducts;
using Propulsion::SolidRocketMotor::critical_pressure_ratio;
using Propulsion::SolidRocketMotor::EndBurningGrain;
using Propulsion::SolidRocketMotor::ErosiveBurning;
using Propulsion::SolidRocketMotor::erosive_burning_rate;
using Propulsion::SolidRocketMotor::flow_function;
using Propulsion::SolidRocketMotor::forward_restriction_mass_flow;
using Propulsion::SolidRocketMotor::ideal_thrust;
using Propulsion::SolidRocketMotor::OpeningSchedule;
using Propulsion::SolidRocketMotor::PintleValve;
using Propulsion::SolidRocketMotor::restriction_mass_flow;
using Propulsion::SolidRocketMotor::SaintRobertBurningRate;
using Propulsion::SolidRocketMotor::static_to_stagnation_pressure_ratio;
using Propulsion::SolidRocketMotor::supersonic_exit_mach;
using Propulsion::SolidRocketMotor::TemperatureSensitivity;
using Propulsion::SolidRocketMotor::TubularGrain;

namespace GoogleUnitTests
{
namespace Propulsion
{
namespace SolidRocketMotor
{

//------------------------------------------------------------------------------
/// Gamma(1.4) = 0.6847 and p*/p_0 = 0.5283 for air are the textbook values
/// (Hill & Peterson Eq. 3.14; Sutton Eq. 3-20).
//------------------------------------------------------------------------------
TEST(CombustionProductsTests, FlowFunctionAndCriticalRatioForAir)
{
  EXPECT_NEAR(flow_function(1.4), 0.684731, 1.0e-6);
  EXPECT_NEAR(critical_pressure_ratio(1.4), 0.528282, 1.0e-6);
}

TEST(CombustionProductsTests, CharacteristicVelocityRoundTrips)
{
  const auto products = CombustionProducts<double>::from_characteristic_velocity(
    1298.448, 1672.039, 1.25);
  EXPECT_NEAR(products.characteristic_velocity(), 1298.448, 1.0e-9);
  EXPECT_NEAR(
    products.specific_heat_at_constant_pressure() -
      products.specific_heat_at_constant_volume(),
    products.specific_gas_constant(),
    1.0e-12);
}

//------------------------------------------------------------------------------
/// sigma_p = d ln a / dT_b: check both laws by central difference, and that
/// the Williams law with T_e = T_ref + 1 / sigma matches the exponential law
/// to first order at T_ref.
//------------------------------------------------------------------------------
TEST(BurningRateTests, TemperatureLawsHaveTheirStatedSigmaP)
{
  const double sigma {0.002};
  const double reference {294.15};
  const auto exponential = TemperatureSensitivity<double>::exponential(
    sigma, reference);
  const auto williams = TemperatureSensitivity<double>::explosion_temperature(
    reference + 1.0 / sigma, reference);

  EXPECT_DOUBLE_EQ(exponential.coefficient_factor(reference), 1.0);
  EXPECT_DOUBLE_EQ(williams.coefficient_factor(reference), 1.0);

  const double dt {1.0e-3};
  for (const double t : {250.0, 294.15, 330.0})
  {
    for (const auto& law : {exponential, williams})
    {
      const double numerical {
        (std::log(law.coefficient_factor(t + dt)) -
          std::log(law.coefficient_factor(t - dt))) / (2.0 * dt)};
      EXPECT_NEAR(numerical, law.sigma_p(t), 1.0e-9);
    }
  }
  EXPECT_NEAR(williams.sigma_p(reference), sigma, 1.0e-15);
}

TEST(BurningRateTests, SaintRobertIsAPowerLaw)
{
  const SaintRobertBurningRate<double> law {
    3.0e-5,
    0.35,
    TemperatureSensitivity<double>::exponential(0.0, 294.15)};
  EXPECT_DOUBLE_EQ(law.rate(0.0, 294.15), 0.0);
  const double ratio {law.rate(8.0e6, 294.15) / law.rate(4.0e6, 294.15)};
  EXPECT_NEAR(ratio, std::pow(2.0, 0.35), 1.0e-14);
}

//------------------------------------------------------------------------------
/// The erosive root satisfies Sutton Eq. 12-17 with G = rho_b A_b r / A_p,
/// exceeds r_0, and reduces to r_0 when alpha = 0.
//------------------------------------------------------------------------------
TEST(BurningRateTests, ErosiveRootSatisfiesLenoirRobillard)
{
  const ErosiveBurning<double> erosive {2.0e-5, 53.0};
  const double r0 {0.0068};
  const double density {1760.0};
  const double burning_area {0.314};
  const double port_area {0.00785};
  const double diameter {0.1};

  const double r {erosive_burning_rate(
    r0, erosive, density, burning_area, port_area, diameter)};
  const double mass_flux {density * burning_area * r / port_area};
  const double right_side {
    r0 + erosive.alpha_ * std::pow(mass_flux, 0.8) * std::pow(diameter, -0.2) *
      std::exp(-erosive.beta_ * r * density / mass_flux)};
  EXPECT_GT(r, r0);
  EXPECT_NEAR(r, right_side, 1.0e-13 * r);

  const ErosiveBurning<double> none {0.0, 53.0};
  EXPECT_DOUBLE_EQ(
    erosive_burning_rate(r0, none, density, burning_area, port_area, diameter),
    r0);
}

//------------------------------------------------------------------------------
/// dV/dy = A_b (solid-ballistics.tex eq. sb-volume-rate) for every grain, by
/// central difference; A_b = 0 and V = case + free volume after burnout.
//------------------------------------------------------------------------------
template <typename Grain>
void expect_volume_rate_is_burning_area(const Grain& grain)
{
  const double h {1.0e-7};
  for (int i {1}; i < 20; ++i)
  {
    const double y {grain.web() * static_cast<double>(i) / 20.0};
    const double derivative {
      (grain.gas_volume(y + h) - grain.gas_volume(y - h)) / (2.0 * h)};
    EXPECT_NEAR(derivative, grain.burning_area(y), 1.0e-6 * grain.burning_area(y));
  }
  EXPECT_DOUBLE_EQ(grain.burning_area(grain.web()), 0.0);
  EXPECT_DOUBLE_EQ(grain.propellant_volume(grain.web()), 0.0);
}

TEST(GrainTests, VolumeRateEqualsBurningAreaForEveryGrain)
{
  expect_volume_rate_is_burning_area(
    TubularGrain<double>{0.05, 0.10, 1.0, true, 0.003});
  expect_volume_rate_is_burning_area(
    TubularGrain<double>{0.05, 0.10, 1.0, false, 0.003});
  // Short grain: the ends burn through before the web does.
  expect_volume_rate_is_burning_area(
    TubularGrain<double>{0.05, 0.30, 0.2, false, 0.003});
  expect_volume_rate_is_burning_area(EndBurningGrain<double>{0.1456, 0.02, 0.0015});
}

TEST(GrainTests, WebIsTheShorterBurnPath)
{
  EXPECT_DOUBLE_EQ((TubularGrain<double>{0.05, 0.10, 1.0, true, 0.0}.web()), 0.05);
  EXPECT_DOUBLE_EQ((TubularGrain<double>{0.05, 0.30, 0.2, false, 0.0}.web()), 0.1);
}

//------------------------------------------------------------------------------
/// Restriction flow: continuous at the critical ratio, zero at equal pressure,
/// antisymmetric under exchange of sides at equal temperatures.
//------------------------------------------------------------------------------
TEST(CompressibleFlowTests, RestrictionFlowIsContinuousAndAntisymmetric)
{
  const CombustionProducts<double> gas {1672.0, 0.019, 1.25};
  const double critical {critical_pressure_ratio(1.25)};
  const double p_u {5.0e6};
  const double below {forward_restriction_mass_flow(
    gas, 1.0e-4, p_u, 1672.0, p_u * critical * (1.0 - 1.0e-12))};
  const double above {forward_restriction_mass_flow(
    gas, 1.0e-4, p_u, 1672.0, p_u * critical * (1.0 + 1.0e-12))};
  EXPECT_NEAR(below, above, 1.0e-9 * below);
  EXPECT_DOUBLE_EQ(
    forward_restriction_mass_flow(gas, 1.0e-4, p_u, 1672.0, p_u), 0.0);
  EXPECT_DOUBLE_EQ(
    restriction_mass_flow(gas, 1.0e-4, 3.0e6, 1500.0, 4.0e6, 1500.0),
    -restriction_mass_flow(gas, 1.0e-4, 4.0e6, 1500.0, 3.0e6, 1500.0));
}

TEST(CompressibleFlowTests, SupersonicExitMachInvertsAreaRatio)
{
  for (const double gamma : {1.15, 1.25, 1.4})
  {
    for (const double epsilon : {1.0, 1.5, 4.0, 25.0, 200.0})
    {
      const double mach {supersonic_exit_mach(epsilon, gamma)};
      EXPECT_GE(mach, 1.0);
      EXPECT_NEAR(area_ratio_at_mach(mach, gamma), epsilon, 1.0e-12 * epsilon);
    }
  }
}

//------------------------------------------------------------------------------
/// Sutton Eq. 3-30 against the control-volume thrust m_dot v_e + (p_e - p_a)
/// A_e computed independently from the exit Mach number.
//------------------------------------------------------------------------------
TEST(CompressibleFlowTests, ThrustCoefficientMatchesMomentumPlusPressure)
{
  const CombustionProducts<double> gas {3300.0, 0.029, 1.18};
  const double p_c {7.0e6};
  const double a_t {1.0e-3};
  const double gamma {gas.heat_capacity_ratio()};
  for (const double epsilon : {1.0, 3.0, 8.0, 40.0})
  {
    for (const double p_a : {0.0, 101325.0})
    {
      const double mach {supersonic_exit_mach(epsilon, gamma)};
      const double exit_pressure {
        p_c * static_to_stagnation_pressure_ratio(mach, gamma)};
      const double exit_temperature {
        gas.stagnation_temperature() /
          (1.0 + (gamma - 1.0) / 2.0 * mach * mach)};
      const double exit_velocity {
        mach * std::sqrt(gamma * gas.specific_gas_constant() * exit_temperature)};
      const double mass_flow {forward_restriction_mass_flow(
        gas, a_t, p_c, gas.stagnation_temperature(), p_a)};
      const double control_volume {
        mass_flow * exit_velocity + (exit_pressure - p_a) * epsilon * a_t};
      EXPECT_NEAR(
        ideal_thrust(gas, 1.0, a_t, epsilon * a_t, p_c,
          gas.stagnation_temperature(), p_a),
        control_volume,
        1.0e-10 * control_volume);
    }
  }
}

TEST(CompressibleFlowTests, ThrustIsContinuousAtUnchokingForSonicExit)
{
  const CombustionProducts<double> gas {1672.0, 0.019, 1.25};
  const double p_a {101325.0};
  const double p_c {p_a / critical_pressure_ratio(1.25)};
  const double choked {ideal_thrust(
    gas, 1.0, 1.0e-4, 1.0e-4, p_c * (1.0 + 1.0e-12), 1672.0, p_a)};
  const double subsonic {ideal_thrust(
    gas, 1.0, 1.0e-4, 1.0e-4, p_c * (1.0 - 1.0e-12), 1672.0, p_a)};
  EXPECT_NEAR(choked, subsonic, 1.0e-8 * choked);
}

//------------------------------------------------------------------------------
/// Pintle area A(x) = pi R^2 [1 - (1 - x)^2]: closed is zero, open is pi R^2,
/// increasing; schedules interpolate linearly and hold at the ends.
//------------------------------------------------------------------------------
TEST(PintleValveTests, AreaAndSchedule)
{
  const PintleValve<double> valve {
    0.01,
    4.0e-4,
    0.95,
    OpeningSchedule<double>{{{0.4, 1.0}, {0.5, 0.0}}}};
  EXPECT_DOUBLE_EQ(valve.flow_area(0.0), 0.0);
  EXPECT_DOUBLE_EQ(valve.flow_area(1.0), std::numbers::pi * 1.0e-4);
  EXPECT_NEAR(valve.flow_area(0.5), 0.75 * std::numbers::pi * 1.0e-4, 1.0e-18);
  double previous {-1.0};
  for (int i {0}; i <= 100; ++i)
  {
    const double area {valve.flow_area(static_cast<double>(i) / 100.0)};
    EXPECT_GT(area, previous);
    previous = area;
  }
  EXPECT_DOUBLE_EQ(valve.schedule().opening_at(0.0), 1.0);
  EXPECT_DOUBLE_EQ(valve.schedule().opening_at(0.45), 0.5);
  EXPECT_DOUBLE_EQ(valve.schedule().opening_at(9.0), 0.0);
}

} // namespace SolidRocketMotor
} // namespace Propulsion
} // namespace GoogleUnitTests
