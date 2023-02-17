use crate::algebra::modules::vectors::RingOperations;

use std::ops::{Add, Sub, Index, Mul, Neg};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VectorPart<T: RingOperations>
{
  data: [T; 3],
}

impl<T: RingOperations> Add for VectorPart<T>
{
  // Self is the type of the current object.
  type Output = Self;

  fn add(self, other: Self) -> Self
  {
    Self {data: [
      self.data[0] + other.data[0],
      self.data[1] + other.data[1],
      self.data[2] + other.data[2]]}
  }
}

impl<T: RingOperations> Sub for VectorPart<T>
{
  // Self is the type of the current object.
  type Output = Self;

  fn sub(self, other: Self) -> Self
  {
    Self {data: [
      self.data[0] - other.data[0],
      self.data[1] - other.data[1],
      self.data[2] - other.data[2]]}
  }
}

impl<T: RingOperations> Index<usize> for VectorPart<T>
{
  // Self is the type of the current object.
  type Output = Self;

  fn index(&self, index_value: usize) -> &Self::Output
  {
    &self.data[index_value - 1]
  }
}

impl<T: RingOperations> Mul<T> for VectorPart<T>
{
  // Self is the type of the current object.
  type Output = Self;

  fn mul(self: Self, rhs: T) -> Self::Output
  {
    Self {data: [
      self.data[0] * rhs,
      self.data[1] * rhs,
      self.data[2] * rhs]}
  }
}

impl<T: RingOperations> Mul<VectorPart<T>> for (T, )
{
  // Self is the type of the current object.
  type Output = VectorPart<T>;

  fn mul(self: Self, rhs: VectorPart<T>) -> Self::Output
  {
    Self {data: [
      rhs.data[0] * self.0,
      rhs.data[1] * self.0,
      rhs.data[2] * self.0]}
  }
}

impl<T: RingOperations> Neg for VectorPart<T>
{
  // Self is the type of the current object.
  type Output = Self;

  fn neg(self) -> Self::Output
  {
    Self {data: [-self.data[0], -self.data[1], -self.data[2]]}
  }
}

impl<T: RingOperations> VectorPart<T>
{
  fn cross_product(self, rhs: Self) -> Self
  {
    Self {data: [
      self.data[1] * rhs.data[2] - self.data[2] * rhs.data[1],
      self.data[2] * rhs.data[0] - self.data[0] * rhs.data[2],
      self.data[0] * rhs.data[1] - self.data[1] * rhs.data[0]
      ]}
  }

  fn dot_product(self, rhs: Self) -> T
  {
    self.data[0] * rhs.data[0] +
      self.data[1] * rhs.data[1] +
      self.data[2] * rhs.data[2]
  }

  fn norm(self) -> T
  {
    self.data[0] * self.data[0] +
      self.data[1] * self.data[1] +
      self.data[2] * self.data[2]
  }
}
