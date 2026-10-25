//! The small linear algebra the rigid body needs: 3-vectors, 3x3 matrices and
//! scalar-first Hamilton quaternions, all as plain arrays.
//!
//! Every sum is accumulated left to right from zero, in the same order as the
//! mathphysics `lie-groups` crate, so that the golden fixtures emitted there
//! are reproduced to the last bit.

pub type Vector3 = [f64; 3];
pub type Matrix3 = [[f64; 3]; 3];
/// Scalar-first Hamilton quaternion `w + x i + y j + z k`.
pub type Quaternion = [f64; 4];

pub const IDENTITY_QUATERNION: Quaternion = [1.0, 0.0, 0.0, 0.0];

pub const IDENTITY_MATRIX: Matrix3 =
  [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub fn dot(a: &Vector3, b: &Vector3) -> f64
{
  (0..3).fold(0.0, |sum, i| sum + a[i] * b[i])
}

pub fn cross(a: &Vector3, b: &Vector3) -> Vector3
{
  [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

pub fn add(a: &Vector3, b: &Vector3) -> Vector3
{
  [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn subtract(a: &Vector3, b: &Vector3) -> Vector3
{
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(a: &Vector3, factor: f64) -> Vector3
{
  [a[0] * factor, a[1] * factor, a[2] * factor]
}

pub fn maximum_norm(a: &Vector3) -> f64
{
  a.iter().fold(0.0_f64, |largest, value| largest.max(value.abs()))
}

pub fn is_finite_vector(a: &Vector3) -> bool
{
  a.iter().all(|value| value.is_finite())
}

pub fn matrix_vector_product(matrix: &Matrix3, vector: &Vector3) -> Vector3
{
  core::array::from_fn(|i| {
    (0..3).fold(0.0, |sum, j| sum + matrix[i][j] * vector[j])
  })
}

pub fn matrix_product(left: &Matrix3, right: &Matrix3) -> Matrix3
{
  core::array::from_fn(|i| {
    core::array::from_fn(|j| {
      (0..3).fold(0.0, |sum, k| sum + left[i][k] * right[k][j])
    })
  })
}

pub fn trace(matrix: &Matrix3) -> f64
{
  (0..3).fold(0.0, |sum, i| sum + matrix[i][i])
}

pub fn determinant(m: &Matrix3) -> f64
{
  m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// The matrix of `v ×`, so that `hat(v) u = v × u`.
pub fn hat(v: &Vector3) -> Matrix3
{
  [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

/// Hamilton product `a ⊗ b = (a_w b_w − a_v·b_v, a_w b_v + b_w a_v + a_v×b_v)`.
pub fn hamilton_product(a: &Quaternion, b: &Quaternion) -> Quaternion
{
  let a_vector = [a[1], a[2], a[3]];
  let b_vector = [b[1], b[2], b[3]];
  let vector_cross = cross(&a_vector, &b_vector);
  [
    a[0] * b[0] - dot(&a_vector, &b_vector),
    a[0] * b[1] + b[0] * a[1] + vector_cross[0],
    a[0] * b[2] + b[0] * a[2] + vector_cross[1],
    a[0] * b[3] + b[0] * a[3] + vector_cross[2],
  ]
}

pub fn quaternion_norm(q: &Quaternion) -> f64
{
  q[0].hypot(q[1]).hypot(q[2]).hypot(q[3])
}

pub fn quaternion_norm_squared(q: &Quaternion) -> f64
{
  q.iter().fold(0.0, |sum, value| sum + value * value)
}

/// `q / |q|`, or `None` if `|q|` is zero or not finite. The sign is kept, so a
/// trajectory never jumps between the two lifts `±q` of one rotation.
pub fn normalized_quaternion(q: &Quaternion) -> Option<Quaternion>
{
  let norm = quaternion_norm(q);
  if !norm.is_finite() || norm == 0.0
  {
    return None;
  }
  Some([q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm])
}

/// The rotation `R(q) = (w² − v·v) 1 + 2w [v]× + 2 v vᵀ` that the unit
/// quaternion `q = (w, v)` applies by `x ↦ q ⊗ (0, x) ⊗ q*`, i.e. body
/// components to inertial components.
pub fn rotation_matrix(q: &Quaternion) -> Matrix3
{
  let vector = [q[1], q[2], q[3]];
  let skew = hat(&vector);
  let scalar = q[0] * q[0] - dot(&vector, &vector);
  let twice_scalar_part = 2.0 * q[0];
  core::array::from_fn(|i| {
    core::array::from_fn(|j| {
      let diagonal = IDENTITY_MATRIX[i][j] * scalar;
      (diagonal + skew[i][j] * twice_scalar_part)
        + 2.0 * vector[i] * vector[j]
    })
  })
}

#[cfg(test)]
mod tests
{
  use super::*;

  #[test]
  fn rotation_matrix_is_the_sandwich_product()
  {
    let q = normalized_quaternion(&[0.8, -0.3, 0.4, 0.2]).unwrap();
    let x = [0.7, -1.1, 0.25];
    let conjugate = [q[0], -q[1], -q[2], -q[3]];
    let sandwich = hamilton_product(
      &hamilton_product(&q, &[0.0, x[0], x[1], x[2]]),
      &conjugate);
    let rotated = matrix_vector_product(&rotation_matrix(&q), &x);
    for i in 0..3
    {
      assert!((sandwich[i + 1] - rotated[i]).abs() < 1.0e-15);
    }
    assert!(sandwich[0].abs() < 1.0e-15);
  }

  #[test]
  fn hat_is_the_cross_product()
  {
    let v = [0.3, -0.2, 0.9];
    let u = [-1.0, 0.5, 0.25];
    assert_eq!(matrix_vector_product(&hat(&v), &u), cross(&v, &u));
  }
}
