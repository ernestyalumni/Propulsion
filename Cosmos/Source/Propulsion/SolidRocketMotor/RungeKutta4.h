//------------------------------------------------------------------------------
/// \file RungeKutta4.h
/// \brief Classical fourth-order Runge-Kutta step on a fixed-size state, with
///   the tableau named.
///
/// Fixed step so the C++ and Rust twins take identical steps (derivation note,
/// section 8). RK4 preserves every linear invariant l . y with l . f == 0 to
/// round-off, which the mass-conservation tests rely on.
/// Rust twin: cosmos_propulsion::solid_rocket_motor::runge_kutta_4.
//------------------------------------------------------------------------------
#ifndef PROPULSION_SOLID_ROCKET_MOTOR_RUNGE_KUTTA_4_H
#define PROPULSION_SOLID_ROCKET_MOTOR_RUNGE_KUTTA_4_H

#include <array>
#include <concepts>
#include <cstddef>

namespace Propulsion
{
namespace SolidRocketMotor
{

template <std::floating_point Field = double>
struct RungeKutta4Tableau
{
  // Nodes c_2, c_3, c_4 (c_1 = 0) and the nonzero a_{i,i-1}.
  static constexpr Field c2 {static_cast<Field>(0.5)};
  static constexpr Field c3 {static_cast<Field>(0.5)};
  static constexpr Field c4 {static_cast<Field>(1)};
  static constexpr Field a21 {static_cast<Field>(0.5)};
  static constexpr Field a32 {static_cast<Field>(0.5)};
  static constexpr Field a43 {static_cast<Field>(1)};
  // Weights b_i.
  static constexpr Field b1 {static_cast<Field>(1) / static_cast<Field>(6)};
  static constexpr Field b2 {static_cast<Field>(1) / static_cast<Field>(3)};
  static constexpr Field b3 {static_cast<Field>(1) / static_cast<Field>(3)};
  static constexpr Field b4 {static_cast<Field>(1) / static_cast<Field>(6)};
};

/// One step y(t) -> y(t + h) of dy/dt = f(t, y).
template <std::floating_point Field, std::size_t N, typename Derivatives>
std::array<Field, N> runge_kutta_4_step(
  const Derivatives& f,
  const Field t,
  const std::array<Field, N>& y,
  const Field h)
{
  using T = RungeKutta4Tableau<Field>;
  std::array<Field, N> stage {};

  const std::array<Field, N> k1 {f(t, y)};
  for (std::size_t i {0}; i < N; ++i)
  {
    stage[i] = y[i] + h * T::a21 * k1[i];
  }
  const std::array<Field, N> k2 {f(t + T::c2 * h, stage)};
  for (std::size_t i {0}; i < N; ++i)
  {
    stage[i] = y[i] + h * T::a32 * k2[i];
  }
  const std::array<Field, N> k3 {f(t + T::c3 * h, stage)};
  for (std::size_t i {0}; i < N; ++i)
  {
    stage[i] = y[i] + h * T::a43 * k3[i];
  }
  const std::array<Field, N> k4 {f(t + T::c4 * h, stage)};

  std::array<Field, N> next {};
  for (std::size_t i {0}; i < N; ++i)
  {
    next[i] = y[i] + h * (T::b1 * k1[i] + T::b2 * k2[i] + T::b3 * k3[i] +
      T::b4 * k4[i]);
  }
  return next;
}

/// Number of bisection halvings of the sub-step when landing on an event;
/// 64 halvings take any double step below its unit round-off.
inline constexpr int event_bisection_iterations {64};

/// One step of size h that does not integrate across the moment the
/// increasing component y[component] reaches threshold. Burnout (A_b drops to
/// zero at y = web) is a discontinuity of the right-hand side, and a step that
/// straddles it would burn propellant that is not there. If the full step
/// crosses the threshold, bisect for the sub-step s that lands on it, set the
/// component to the threshold exactly, and finish with a step of h - s.
template <std::floating_point Field, std::size_t N, typename Derivatives>
std::array<Field, N> runge_kutta_4_step_stopping_at(
  const Derivatives& f,
  const Field t,
  const std::array<Field, N>& y,
  const Field h,
  const std::size_t component,
  const Field threshold)
{
  const std::array<Field, N> full {runge_kutta_4_step(f, t, y, h)};
  if (!(y[component] < threshold && full[component] > threshold))
  {
    return full;
  }
  Field lower {static_cast<Field>(0)};
  Field upper {h};
  for (int iteration {0}; iteration < event_bisection_iterations; ++iteration)
  {
    const Field middle {(lower + upper) / static_cast<Field>(2)};
    if (runge_kutta_4_step(f, t, y, middle)[component] < threshold)
    {
      lower = middle;
    }
    else
    {
      upper = middle;
    }
  }
  std::array<Field, N> landed {runge_kutta_4_step(f, t, y, upper)};
  landed[component] = threshold;
  return runge_kutta_4_step(f, t + upper, landed, h - upper);
}

} // namespace SolidRocketMotor
} // namespace Propulsion

#endif // PROPULSION_SOLID_ROCKET_MOTOR_RUNGE_KUTTA_4_H
