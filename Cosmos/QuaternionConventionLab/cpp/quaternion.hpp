#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace cosmos::rotation {

struct Vector3 {
  double x{};
  double y{};
  double z{};
};

using Matrix3 = std::array<double, 9>; // row major

// Explicit contract used throughout this demo:
//   - Hamilton product: i*j=k, j*k=i, k*i=j
//   - scalar first: (w, x, y, z)
//   - active rotation: body/local vector -> inertial/world vector
//   - rotated vector: v_world = q * (0, v_body) * conjugate(q)
struct Quaternion {
  double w{1.0};
  double x{};
  double y{};
  double z{};
};

struct ScalarLastQuaternion {
  double x{};
  double y{};
  double z{};
  double w{1.0};
};

inline double dot(const Vector3& lhs, const Vector3& rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline double magnitude(const Vector3& vector) {
  return std::sqrt(dot(vector, vector));
}

inline Vector3 normalized(const Vector3& vector) {
  const double norm = magnitude(vector);
  if (norm == 0.0) {
    throw std::invalid_argument("rotation axis must be nonzero");
  }
  return {vector.x / norm, vector.y / norm, vector.z / norm};
}

inline double dot(const Quaternion& lhs, const Quaternion& rhs) {
  return lhs.w * rhs.w + lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline Quaternion normalized(const Quaternion& quaternion) {
  const double norm = std::sqrt(dot(quaternion, quaternion));
  if (norm == 0.0) {
    throw std::invalid_argument("quaternion norm must be nonzero");
  }
  return {quaternion.w / norm, quaternion.x / norm,
          quaternion.y / norm, quaternion.z / norm};
}

inline Quaternion conjugate(const Quaternion& quaternion) {
  return {quaternion.w, -quaternion.x, -quaternion.y, -quaternion.z};
}

inline Quaternion negated(const Quaternion& quaternion) {
  return {-quaternion.w, -quaternion.x, -quaternion.y, -quaternion.z};
}

inline Quaternion hamilton_product(const Quaternion& lhs,
                                   const Quaternion& rhs) {
  return {
      lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
      lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
      lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
      lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
  };
}

inline Quaternion from_axis_angle(const Vector3& axis, double angle_radians) {
  const Vector3 unit_axis = normalized(axis);
  const double half_angle = 0.5 * angle_radians;
  const double sine = std::sin(half_angle);
  return {std::cos(half_angle), sine * unit_axis.x,
          sine * unit_axis.y, sine * unit_axis.z};
}

inline Vector3 rotate_active(const Quaternion& rotation,
                             const Vector3& vector) {
  const Quaternion q = normalized(rotation);
  const Quaternion pure{0.0, vector.x, vector.y, vector.z};
  const Quaternion rotated =
      hamilton_product(hamilton_product(q, pure), conjugate(q));
  return {rotated.x, rotated.y, rotated.z};
}

inline Matrix3 to_rotation_matrix(const Quaternion& rotation) {
  const Quaternion q = normalized(rotation);
  const double xx = q.x * q.x;
  const double yy = q.y * q.y;
  const double zz = q.z * q.z;
  const double xy = q.x * q.y;
  const double xz = q.x * q.z;
  const double yz = q.y * q.z;
  const double wx = q.w * q.x;
  const double wy = q.w * q.y;
  const double wz = q.w * q.z;

  return {
      1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
      2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
      2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy),
  };
}

inline ScalarLastQuaternion to_scalar_last(const Quaternion& quaternion) {
  const Quaternion q = normalized(quaternion);
  return {q.x, q.y, q.z, q.w};
}

inline Quaternion from_scalar_last(const ScalarLastQuaternion& quaternion) {
  return normalized({quaternion.w, quaternion.x, quaternion.y, quaternion.z});
}

// The physical attitude error must be sign invariant because q and -q are the
// same SO(3) rotation. Taking abs(dot) chooses the nearer S^3 representative.
inline double physical_rotation_distance(const Quaternion& lhs,
                                         const Quaternion& rhs) {
  const double cosine_half_angle =
      std::clamp(std::abs(dot(normalized(lhs), normalized(rhs))), 0.0, 1.0);
  return 2.0 * std::acos(cosine_half_angle);
}

// Essential before interpolation, filtering residuals, or differencing a time
// history: keep adjacent quaternion samples on the same S^3 hemisphere.
inline Quaternion align_hemisphere(const Quaternion& reference,
                                   const Quaternion& candidate) {
  return dot(normalized(reference), normalized(candidate)) < 0.0
             ? negated(candidate)
             : candidate;
}

} // namespace cosmos::rotation
