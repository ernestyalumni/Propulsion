use crate::algebra::vectors::RingOperations;

use std::ops::{Add, Sub, Index, Mul, Neg};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VectorPart<T: RingOperations>
{
  data: [T; 3],
}