use std::ops::{Add, Sub, Mul, Neg};

pub trait RingOperations:
	Add<Output=Self> +
    Sub<Output=Self> +
    Mul<Output=Self> +
    Copy +
    Neg<Output=Self>
  where Self: std::marker::Sized
{}

impl<T> RingOperations for T
  where T: Add<Output=T> +
    Sub<Output=T> +
    Mul<Output=T> +
    Copy +
    Neg<Output=T> +
{}