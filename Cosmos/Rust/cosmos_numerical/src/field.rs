//! The scalar field the numerics are generic over.
//!
//! `wildrider::algebra::fields::FieldOperations` is the algebraic spine shared
//! with the rest of the repository: a type with `+ - * /` and negation. Numerical
//! methods additionally need an order, square roots, logarithms, powers, and a
//! machine epsilon, which is what `RealField` adds. Only `f64` and `f32`
//! implement it today.

use wildrider::algebra::fields::FieldOperations;

pub trait RealField: FieldOperations + PartialOrd + core::fmt::Debug
{
  fn zero() -> Self;
  fn one() -> Self;
  fn from_f64(value: f64) -> Self;
  fn to_f64(self) -> f64;
  fn square_root(self) -> Self;
  fn natural_logarithm(self) -> Self;
  fn power(self, exponent: Self) -> Self;
  fn absolute_value(self) -> Self;
  fn machine_epsilon() -> Self;
  fn is_finite(self) -> bool;

  fn maximum(self, other: Self) -> Self
  {
    if other > self { other } else { self }
  }

  fn minimum(self, other: Self) -> Self
  {
    if other < self { other } else { self }
  }
}

macro_rules! implement_real_field
{
  ($t:ty) =>
  {
    impl RealField for $t
    {
      fn zero() -> Self { 0.0 }
      fn one() -> Self { 1.0 }
      fn from_f64(value: f64) -> Self { value as $t }
      fn to_f64(self) -> f64 { self as f64 }
      fn square_root(self) -> Self { self.sqrt() }
      fn natural_logarithm(self) -> Self { self.ln() }
      fn power(self, exponent: Self) -> Self { self.powf(exponent) }
      fn absolute_value(self) -> Self { self.abs() }
      fn machine_epsilon() -> Self { <$t>::EPSILON }
      fn is_finite(self) -> bool { <$t>::is_finite(self) }
    }
  };
}

implement_real_field!(f64);
implement_real_field!(f32);
