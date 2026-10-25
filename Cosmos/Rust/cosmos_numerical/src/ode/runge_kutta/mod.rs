//! Runge–Kutta machinery: the classical fixed-step method, and the embedded
//! pairs of Dormand and Prince with Hairer's step-size control and dense
//! output (HNW I §II.4–II.6).

pub mod classical_4;
pub mod dormand_prince_5;
pub mod dormand_prince_853;
pub mod embedded;
pub mod initial_step;
pub mod pi_step_size;
pub mod tolerance;

#[cfg(test)]
mod order_conditions;
#[cfg(test)]
mod hairer_reference_tests;
#[cfg(test)]
mod test_problems;
