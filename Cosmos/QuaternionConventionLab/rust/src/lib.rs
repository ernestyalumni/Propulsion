//! Minimal quaternion kernel with an explicit, testable convention.
//!
//! Contract: Hamilton product, scalar-first `(w, x, y, z)`, active rotation
//! from body/local coordinates to inertial/world coordinates.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    pub fn normalized(self) -> Self {
        let norm = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        assert!(norm > 0.0, "rotation axis must be nonzero");
        Self {
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarLastQuaternion {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Default for Quaternion {
    fn default() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl Quaternion {
    pub fn normalized(self) -> Self {
        let norm = self.dot(self).sqrt();
        assert!(norm > 0.0, "quaternion norm must be nonzero");
        Self {
            w: self.w / norm,
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    pub fn negated(self) -> Self {
        Self {
            w: -self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Hamilton product: `i*j=k`, `j*k=i`, `k*i=j`.
    pub fn hamilton_product(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }

    pub fn from_axis_angle(axis: Vector3, angle_radians: f64) -> Self {
        let axis = axis.normalized();
        let half_angle = 0.5 * angle_radians;
        let sine = half_angle.sin();
        Self {
            w: half_angle.cos(),
            x: sine * axis.x,
            y: sine * axis.y,
            z: sine * axis.z,
        }
    }

    /// Active rotation: `v_world = q * (0,v_body) * conjugate(q)`.
    pub fn rotate_active(self, vector: Vector3) -> Vector3 {
        let q = self.normalized();
        let pure = Self {
            w: 0.0,
            x: vector.x,
            y: vector.y,
            z: vector.z,
        };
        let rotated = q.hamilton_product(pure).hamilton_product(q.conjugate());
        Vector3 {
            x: rotated.x,
            y: rotated.y,
            z: rotated.z,
        }
    }

    pub fn rotation_matrix(self) -> [f64; 9] {
        let q = self.normalized();
        let (xx, yy, zz) = (q.x * q.x, q.y * q.y, q.z * q.z);
        let (xy, xz, yz) = (q.x * q.y, q.x * q.z, q.y * q.z);
        let (wx, wy, wz) = (q.w * q.x, q.w * q.y, q.w * q.z);
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ]
    }

    pub fn to_scalar_last(self) -> ScalarLastQuaternion {
        let q = self.normalized();
        ScalarLastQuaternion {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        }
    }

    pub fn from_scalar_last(q: ScalarLastQuaternion) -> Self {
        Self {
            w: q.w,
            x: q.x,
            y: q.y,
            z: q.z,
        }
        .normalized()
    }

    /// Sign-invariant attitude distance because `q` and `-q` map to the same SO(3) rotation.
    pub fn physical_rotation_distance(self, other: Self) -> f64 {
        let cosine_half_angle = self
            .normalized()
            .dot(other.normalized())
            .abs()
            .clamp(0.0, 1.0);
        2.0 * cosine_half_angle.acos()
    }

    /// Choose the candidate on the reference sample's S^3 hemisphere.
    pub fn align_hemisphere(self, candidate: Self) -> Self {
        if self.normalized().dot(candidate.normalized()) < 0.0 {
            candidate.negated()
        } else {
            candidate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    const EPSILON: f64 = 1.0e-12;

    fn near(lhs: f64, rhs: f64) -> bool {
        (lhs - rhs).abs() <= EPSILON
    }

    #[test]
    fn antipodal_quaternions_have_the_same_rotation_matrix() {
        let q = Quaternion::from_axis_angle(
            Vector3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            0.73,
        );
        let antipode = q.negated();
        for (lhs, rhs) in q
            .rotation_matrix()
            .iter()
            .zip(antipode.rotation_matrix().iter())
        {
            assert!(near(*lhs, *rhs));
        }
        assert!(near(q.physical_rotation_distance(antipode), 0.0));
    }

    #[test]
    fn active_positive_z_quarter_turn_maps_x_to_y() {
        let q = Quaternion::from_axis_angle(
            Vector3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            FRAC_PI_2,
        );
        let result = q.rotate_active(Vector3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        });
        assert!(near(result.x, 0.0));
        assert!(near(result.y, 1.0));
        assert!(near(result.z, 0.0));
    }

    #[test]
    fn scalar_layout_adapter_round_trips() {
        let q = Quaternion::from_axis_angle(
            Vector3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            PI / 3.0,
        );
        let round_trip = Quaternion::from_scalar_last(q.to_scalar_last());
        assert!(near(q.physical_rotation_distance(round_trip), 0.0));
    }

    #[test]
    fn hemisphere_alignment_removes_an_artificial_sign_jump() {
        let reference = Quaternion::from_axis_angle(
            Vector3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            0.4,
        );
        let aligned = reference.align_hemisphere(reference.negated());
        assert!(reference.dot(aligned) > 0.0);
    }
}
