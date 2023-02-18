use std::ops::{Add, Sub, Mul, Div, Neg};

pub trait FieldOperations:
	Add<Output=Self> +
    Sub<Output=Self> +
    Mul<Output=Self> +
    Div<Output=Self> +
    Copy +
    Neg<Output=Self>
  where Self: std::marker::Sized
{}

impl<T> FieldOperations for T
  where T: Add<Output=T> +
    Sub<Output=T> +
    Mul<Output=T> +
    Div<Output=T> +
    Copy +
    Neg<Output=T> +
{}