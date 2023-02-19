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
      self.data[2] + other.data[2],
      self.data[3] + other.data[3],
      self.data[4] + other.data[4],
      self.data[5] + other.data[5],
      self.data[6] + other.data[6],
      self.data[7] + other.data[7],
      self.data[8] + other.data[8]
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
      self.data[2] - other.data[2],
      self.data[3] - other.data[3],
      self.data[4] - other.data[4],
      self.data[5] - other.data[5],
      self.data[6] - other.data[6],
      self.data[7] - other.data[7],
      self.data[8] - other.data[8]
  ]}}
}

// See https://stackoverflow.com/questions/33770989/implementing-the-index-operator-for-matrices-with-multiple-parameters

impl<T: FieldOperations> Index<[usize; 2]> for Rotation3<T>
{
  // Self is the type of the current object.
  type Output = T;

  fn index(&self, indices: [usize; 2]) -> &Self::Output
  {
    &self.data[indices[0] * 3 + indices[1]]
  }
}

impl<T: FieldOperations> IndexMut<[usize; 2]> for Rotation3<T>
{
  fn index_mut(&mut self, indices: [usize; 2]) -> &mut Self::Output
  {
    &mut self.data[indices[0] * 3 + indices[1]]
  }
}

//------------------------------------------------------------------------------
/// \details Compare this matrix multiplication against LAPACK
/// See https://netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html
/// which was from
/// \url https://netlib.org/blas/#_level_3
/// which was from
/// https://github.com/blas-lapack-rs/blas-sys/blob/master/src/lib.rs
//------------------------------------------------------------------------------

impl<T: FieldOperations> Mul<Rotation3<T>> for Rotation3<T>
{
  type Output = Self;

  fn mul(self, b: Self) -> Self
  {
    let mut c : Rotation3<T> = Self {data: self.data};

    for i in 0..3
    {
      for j in 0..3
      {
        c[[i, j]] = self[[i, 0]] * b[[0, j]];
      }
    }

    for i in 0..3
    {
      for k in 1..3
      {
        let temp = self[[i, k]];

        for j in 0..3
        {
          c[[i, j]] = c[[i, j]] + temp * b[[k, j]];
        }
      }
    }

    c
  }
}

impl<T: FieldOperations> Neg for Rotation3<T>
{
  type Output = Self;

  fn neg(self) -> Self
  {
    Self {data: [
      -self.data[0],
      -self.data[1],
      -self.data[2],
      -self.data[3],
      -self.data[4],
      -self.data[5],
      -self.data[6],
      -self.data[7],
      -self.data[8]
    ]}
  }
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn rotation3_multiplies()
  {
    // See https://byjus.com/maths/matrix-multiplication/
    let A = Rotation3::<i32> {data: [12, 8, 4, 3, 17, 14, 9, 8, 10]};
    let B = Rotation3::<i32> {data: [5, 19, 3, 6, 15, 9, 7, 8, 16]};
    let C = A * B;

    assert_eq!(C[[0, 0]], 136);
    assert_eq!(C[[0, 1]], 380);
    assert_eq!(C[[0, 2]], 172);
    assert_eq!(C[[1, 0]], 215);
    assert_eq!(C[[1, 1]], 424);
    assert_eq!(C[[1, 2]], 386);
    assert_eq!(C[[2, 0]], 163);
    assert_eq!(C[[2, 1]], 371);
    assert_eq!(C[[2, 2]], 259);
  }
}