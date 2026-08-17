#include "quaternion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

using cosmos::rotation::Matrix3;
using cosmos::rotation::Quaternion;
using cosmos::rotation::Vector3;

constexpr double kPi = 3.14159265358979323846;
constexpr double kTolerance = 1.0e-12;

bool near(double lhs, double rhs, double tolerance = kTolerance) {
  return std::abs(lhs - rhs) <= tolerance;
}

bool matrices_near(const Matrix3& lhs, const Matrix3& rhs) {
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    if (!near(lhs[index], rhs[index])) {
      return false;
    }
  }
  return true;
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

double radians_to_degrees(double radians) {
  return radians * 180.0 / kPi;
}

void print_vector(const Vector3& vector) {
  std::cout << '(' << vector.x << ", " << vector.y << ", " << vector.z
            << ')';
}

} // namespace

int main() {
  using namespace cosmos::rotation;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Contract: Hamilton, scalar-first (w,x,y,z), active "
               "body-to-world rotation\n\n";

  const Quaternion quarter_turn =
      from_axis_angle({0.0, 0.0, 1.0}, 0.5 * kPi);
  const Quaternion antipode = negated(quarter_turn);
  require(matrices_near(to_rotation_matrix(quarter_turn),
                        to_rotation_matrix(antipode)),
          "q and -q rotation matrices differ");
  require(near(physical_rotation_distance(quarter_turn, antipode), 0.0),
          "q and -q physical error is nonzero");
  std::cout << "PASS  q and -q produce the same direction-cosine matrix\n";
  std::cout << "      physical attitude error = "
            << radians_to_degrees(
                   physical_rotation_distance(quarter_turn, antipode))
            << " deg\n";

  const Vector3 rotated_x = rotate_active(quarter_turn, {1.0, 0.0, 0.0});
  require(near(rotated_x.x, 0.0) && near(rotated_x.y, 1.0) &&
              near(rotated_x.z, 0.0),
          "+90 degrees about +Z did not map +X to +Y");
  std::cout << "PASS  active +90 deg about +Z maps +X to ";
  print_vector(rotated_x);
  std::cout << "\n";

  const Quaternion one_turn = from_axis_angle({0.0, 0.0, 1.0}, 2.0 * kPi);
  const Quaternion two_turns = from_axis_angle({0.0, 0.0, 1.0}, 4.0 * kPi);
  require(near(one_turn.w, -1.0), "360 degrees did not reach -identity");
  require(near(two_turns.w, 1.0), "720 degrees did not return to identity");
  require(near(physical_rotation_distance(one_turn, Quaternion{}), 0.0),
          "360-degree quaternion is not physical identity");
  std::cout << "DEMO  360 deg -> q.w = " << one_turn.w
            << ", but the physical rotation is identity\n";
  std::cout << "DEMO  720 deg -> q.w = " << two_turns.w
            << ", returning to the original SU(2) representative\n";

  const ScalarLastQuaternion scalar_last = to_scalar_last(quarter_turn);
  const Quaternion round_trip = from_scalar_last(scalar_last);
  require(near(physical_rotation_distance(quarter_turn, round_trip), 0.0),
          "scalar-layout adapter did not round-trip");
  std::cout << "PASS  explicit scalar-first <-> scalar-last adapter round-trips\n";

  // Failure injection: bytes documented as [x,y,z,w] are consumed as if they
  // were [w,x,y,z]. This is valid arithmetic with the wrong interface contract.
  const Quaternion layout_mismatch = normalized(
      {scalar_last.x, scalar_last.y, scalar_last.z, scalar_last.w});
  const double layout_error =
      radians_to_degrees(physical_rotation_distance(quarter_turn,
                                                     layout_mismatch));
  require(layout_error > 1.0, "failure injection did not create an error");
  std::cout << "DEMO  scalar-layout mismatch attitude error = " << layout_error
            << " deg\n";

  const Vector3 passive_result =
      rotate_active(conjugate(quarter_turn), {1.0, 0.0, 0.0});
  require(near(passive_result.y, -1.0),
          "passive inversion did not reverse the quarter turn");
  std::cout << "DEMO  active/passive inversion maps +X to ";
  print_vector(passive_result);
  std::cout << " instead\n";

  const Quaternion aligned = align_hemisphere(quarter_turn, antipode);
  require(dot(quarter_turn, aligned) > 0.0,
          "hemisphere alignment left a negative dot product");
  std::cout << "PASS  hemisphere alignment prevents artificial sign jumps\n";

  return 0;
}
