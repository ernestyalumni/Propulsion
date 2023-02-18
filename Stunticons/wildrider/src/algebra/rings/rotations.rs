use crate::algebra::fields::FieldOperations;
use std::ops::{Add, Sub, Index, IndexMut, Mul, Neg};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rotation3<T: FieldOperations>
{
  data: [T; 9],
}

impl<T: FieldOperations> Add for Rotation3<T>
{
  // Self is the type of the current object.
  type Output = Self;

  fn add(self, other: Self) -> Self
  {
    Self {data: [
      self.data[0] + other.data[0],
      self.data[1] + other.data[1],
      self.data[2] + other.data[2]
      self.data[0] + other.data[0],
      self.data[1] + other.data[1],
      self.data[2] + other.data[2]
      self.data[0] + other.data[0],
      self.data[1] + other.data[1],
      self.data[2] + other.data[2]
      ]}
  }
}

impl<T: FieldOperations> Sub for Rotation3<T>
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

impl<T: FieldOperations> Index<usize> for Rotation3<T>
{
  // Self is the type of the current object.
  type Output = T;

  fn index(&self, index_value: usize) -> &Self::Output
  {
    &self.data[index_value - 1]
  }
}
