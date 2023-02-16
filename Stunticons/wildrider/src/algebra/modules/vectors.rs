#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Vector3<T>
{
  data: [T; 3],
}

/*
impl<T> PartialEq for Vector3<T>
{
  fn eq(&self, other: &Self) -> bool
  {
    self.data[0] == other.data[0] && self.data[1] == other.data[1] &&
      self.data[2] == other.data[2]
  }
}
*/

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn vector3_constructs_from_array()
  {
    let x : Vector3<i32> = Vector3::<i32> {data: [1, 2, 3]};

    assert_eq!(x, Vector3::<i32> {data: [1, 2, 3]});
  }
}