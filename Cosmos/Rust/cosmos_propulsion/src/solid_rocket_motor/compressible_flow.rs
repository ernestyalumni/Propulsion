//! Isentropic flow through a restriction (choked, subsonic, reversed), the
//! supersonic exit Mach number for an area ratio, and ideal thrust.
//!
//! Choked: Sutton 9e Eq. 3-24 (p. 59) = Hill & Peterson Eq. 3.14 (p. 71).
//! Subsonic: Sutton Eq. 3-25 solved for the flux. Thrust coefficient: Sutton
//! Eq. 3-30 (p. 62). Exit velocity: Sutton Eq. 3-16.
//! C++ twin: `Propulsion/SolidRocketMotor/CompressibleFlow.h`.

use cosmos_numerical::field::RealField;

use super::combustion_products::{critical_pressure_ratio, flow_function, CombustionProducts};

/// Mass flow from stagnation (p_u, T_u) to static p_d <= p_u through C_d A.
pub fn forward_restriction_mass_flow<T: RealField>(
  products: &CombustionProducts<T>,
  effective_area: T,
  upstream_pressure: T,
  upstream_temperature: T,
  downstream_pressure: T,
) -> T
{
  if effective_area <= T::zero() || upstream_pressure <= T::zero()
  {
    return T::zero();
  }
  let gamma = products.heat_capacity_ratio();
  let gas_constant = products.specific_gas_constant();
  let ratio = downstream_pressure / upstream_pressure;
  if ratio <= critical_pressure_ratio(gamma)
  {
    return effective_area * upstream_pressure * flow_function(gamma)
      / (gas_constant * upstream_temperature).square_root();
  }
  let one = T::one();
  let two = T::from_f64(2.0);
  let bracket = ratio.power(two / gamma) - ratio.power((gamma + one) / gamma);
  let positive_bracket = if bracket > T::zero() { bracket } else { T::zero() };
  effective_area
    * upstream_pressure
    * (two * gamma / ((gamma - one) * gas_constant * upstream_temperature) * positive_bracket).square_root()
}

/// Signed flow from side a to side b; reversed flow uses side b's temperature.
pub fn restriction_mass_flow<T: RealField>(
  products: &CombustionProducts<T>,
  effective_area: T,
  pressure_a: T,
  temperature_a: T,
  pressure_b: T,
  temperature_b: T,
) -> T
{
  if pressure_a >= pressure_b
  {
    return forward_restriction_mass_flow(products, effective_area, pressure_a, temperature_a, pressure_b);
  }
  -forward_restriction_mass_flow(products, effective_area, pressure_b, temperature_b, pressure_a)
}

/// A / A* at Mach M.
pub fn area_ratio_at_mach<T: RealField>(mach: T, heat_capacity_ratio: T) -> T
{
  let gamma = heat_capacity_ratio;
  let one = T::one();
  let two = T::from_f64(2.0);
  (one / mach)
    * ((two / (gamma + one)) * (one + (gamma - one) / two * mach * mach))
      .power((gamma + one) / (two * (gamma - one)))
}

pub const EXIT_MACH_RELATIVE_TOLERANCE: f64 = 1.0e-15;
pub const EXIT_MACH_MAXIMUM_ITERATIONS: usize = 200;

/// Supersonic root M >= 1 of A / A* = expansion_ratio (>= 1).
pub fn supersonic_exit_mach<T: RealField>(expansion_ratio: T, heat_capacity_ratio: T) -> T
{
  let two = T::from_f64(2.0);
  let tolerance = T::from_f64(EXIT_MACH_RELATIVE_TOLERANCE);
  let mut lower = T::one();
  if expansion_ratio <= lower
  {
    return lower;
  }
  let mut upper = two;
  while area_ratio_at_mach(upper, heat_capacity_ratio) < expansion_ratio
  {
    lower = upper;
    upper = two * upper;
  }
  for _ in 0..EXIT_MACH_MAXIMUM_ITERATIONS
  {
    let middle = (lower + upper) / two;
    if area_ratio_at_mach(middle, heat_capacity_ratio) < expansion_ratio
    {
      lower = middle;
    }
    else
    {
      upper = middle;
    }
    if upper - lower <= tolerance * upper
    {
      break;
    }
  }
  (lower + upper) / two
}

/// p / p_0 = (1 + (gamma - 1) M^2 / 2)^(-gamma / (gamma - 1)).
pub fn static_to_stagnation_pressure_ratio<T: RealField>(mach: T, heat_capacity_ratio: T) -> T
{
  let gamma = heat_capacity_ratio;
  let one = T::one();
  let two = T::from_f64(2.0);
  (one + (gamma - one) / two * mach * mach).power(-gamma / (gamma - one))
}

/// C_F p_c C_d A_t when choked, m_dot v_e with p_e = p_a otherwise; zero at
/// or below ambient. Over-expanded separation is not modeled.
pub fn ideal_thrust<T: RealField>(
  products: &CombustionProducts<T>,
  discharge_coefficient: T,
  throat_area: T,
  exit_area: T,
  chamber_pressure: T,
  chamber_temperature: T,
  ambient_pressure: T,
) -> T
{
  if throat_area <= T::zero() || chamber_pressure <= ambient_pressure
  {
    return T::zero();
  }
  let gamma = products.heat_capacity_ratio();
  let one = T::one();
  let two = T::from_f64(2.0);
  let ambient_ratio = ambient_pressure / chamber_pressure;
  if ambient_ratio <= critical_pressure_ratio(gamma)
  {
    let expansion_ratio = if exit_area > throat_area { exit_area / throat_area } else { one };
    let exit_ratio = static_to_stagnation_pressure_ratio(supersonic_exit_mach(expansion_ratio, gamma), gamma);
    let momentum = (two * gamma * gamma / (gamma - one)
      * (two / (gamma + one)).power((gamma + one) / (gamma - one))
      * (one - exit_ratio.power((gamma - one) / gamma)))
    .square_root();
    let thrust_coefficient = momentum + (exit_ratio - ambient_ratio) * expansion_ratio;
    return thrust_coefficient * chamber_pressure * discharge_coefficient * throat_area;
  }
  let mass_flow = forward_restriction_mass_flow(
    products,
    discharge_coefficient * throat_area,
    chamber_pressure,
    chamber_temperature,
    ambient_pressure,
  );
  let exit_velocity = (two * gamma / (gamma - one)
    * products.specific_gas_constant()
    * chamber_temperature
    * (one - ambient_ratio.power((gamma - one) / gamma)))
  .square_root();
  mass_flow * exit_velocity
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn restriction_flow_is_continuous_and_antisymmetric()
  {
    let gas = CombustionProducts::with_si_gas_constant(1672.0_f64, 0.019, 1.25).unwrap();
    let critical = critical_pressure_ratio(1.25_f64);
    let p_u = 5.0e6;
    let below = forward_restriction_mass_flow(&gas, 1.0e-4, p_u, 1672.0, p_u * critical * (1.0 - 1.0e-12));
    let above = forward_restriction_mass_flow(&gas, 1.0e-4, p_u, 1672.0, p_u * critical * (1.0 + 1.0e-12));
    assert!((below - above).abs() < 1.0e-9 * below);
    assert_eq!(forward_restriction_mass_flow(&gas, 1.0e-4, p_u, 1672.0, p_u), 0.0);
    assert_eq!(
      restriction_mass_flow(&gas, 1.0e-4, 3.0e6, 1500.0, 4.0e6, 1500.0),
      -restriction_mass_flow(&gas, 1.0e-4, 4.0e6, 1500.0, 3.0e6, 1500.0)
    );
  }

  #[test]
  fn thrust_coefficient_matches_momentum_plus_pressure()
  {
    let gas = CombustionProducts::with_si_gas_constant(3300.0_f64, 0.029, 1.18).unwrap();
    let (p_c, a_t, gamma): (f64, f64, f64) = (7.0e6, 1.0e-3, 1.18);
    for epsilon in [1.0_f64, 3.0, 8.0, 40.0]
    {
      for p_a in [0.0, 101325.0]
      {
        let mach = supersonic_exit_mach(epsilon, gamma);
        assert!((area_ratio_at_mach(mach, gamma) - epsilon).abs() < 1.0e-12 * epsilon);
        let exit_pressure = p_c * static_to_stagnation_pressure_ratio(mach, gamma);
        let exit_temperature = 3300.0 / (1.0 + (gamma - 1.0) / 2.0 * mach * mach);
        let exit_velocity = mach * (gamma * gas.specific_gas_constant() * exit_temperature).sqrt();
        let mass_flow = forward_restriction_mass_flow(&gas, a_t, p_c, 3300.0, p_a);
        let control_volume = mass_flow * exit_velocity + (exit_pressure - p_a) * epsilon * a_t;
        let thrust = ideal_thrust(&gas, 1.0, a_t, epsilon * a_t, p_c, 3300.0, p_a);
        assert!((thrust - control_volume).abs() < 1.0e-10 * control_volume);
      }
    }
  }
}
