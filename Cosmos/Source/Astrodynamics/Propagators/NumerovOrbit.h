//------------------------------------------------------------------------------
/// \file   NumerovOrbit.h
/// \brief  Orbit propagator: Numerov PECECE + TwoBodyAcceleration + Vector3.
///
/// \details Wraps Numerical::ODE::StormerMethods::NumerovStep (order-4
///   position, order-2 velocity) with the Cosmos algebra and astrodynamics
///   libraries to produce a clean, reusable orbit propagation object.
///
/// Architecture
/// ─────────────────────────────────────────────────────────────────────────
///   OrbitalState<Field>     — plain data: time, position Vector3, velocity
///                             Vector3.  Returned by get_state().
///
///   NumerovOrbit<Field>     — propagation engine.  Accepts any combination
///                             of TwoBodyAcceleration::AccelerationFunctors
///                             (NewtonsGravitation, J2Perturbation, …) packed
///                             into a TotalAcceleration, plus initial
///                             conditions and a step size.
///
///   AccelerationAdapter     — inner struct that bridges TotalAcceleration's
///                             call signature  (const AccelerationInputs&)
///                             to NumerovStep's signature  (Field t, const V3&).
///                             Time is unused since gravitational fields here
///                             are autonomous (no time-varying forces).
///
/// Usage
/// ─────────────────────────────────────────────────────────────────────────
///   // Build acceleration model
///   TotalAcceleration<double> accel{};
///   accel.add(NewtonsGravitation<double>{MU_EARTH});
///   accel.add(J2Perturbation<double>{MU_EARTH, J2, R_EARTH});
///
///   // Initial conditions
///   OrbitalState<double> s0{0.0, r0_vec, v0_vec};
///
///   // Create propagator (startup uses StormerStep sub-steps internally)
///   NumerovOrbit<double> prop{std::move(accel), s0, dt};
///
///   // Advance one step at a time
///   prop.step();
///   auto s = prop.get_state();  // s.t, s.r, s.v
///
/// Startup accuracy note
/// ─────────────────────────────────────────────────────────────────────────
///   NumerovStep::startup() uses StormerStep with n_sub=128 leapfrog
///   sub-steps to produce y_1.  For the typical orbital dt=60 s this gives
///   a startup error of O(dt/128)^2 ≈ 0.22 s^2, well below the O(dt^5)
///   requirement of HNW Theorem 10.6 for full order-4 global convergence.
///   Increase startup_substeps for larger dt.
///
/// References
/// ─────────────────────────────────────────────────────────────────────────
///   - Hairer, Nørsett, Wanner, "Solving ODEs I", §III.10 (Numerov)
///   - Curtis, "Orbital Mechanics for Engineering Students", §4 (J2)
///   - StormerRule.tex (Cosmos docs)
//------------------------------------------------------------------------------

#ifndef ASTRODYNAMICS_PROPAGATORS_NUMEROV_ORBIT_H
#define ASTRODYNAMICS_PROPAGATORS_NUMEROV_ORBIT_H

#include "Algebra/Modules/Vectors/Vector3.h"
#include "Astrodynamics/TwoBodyAcceleration.h"
#include "Numerical/ODE/StormerMethods/NumerovStep.h"
#include "Numerical/ODE/StormerMethods/NumerovState.h"

#include <concepts>
#include <cstddef>

namespace Astrodynamics
{
namespace Propagators
{

//------------------------------------------------------------------------------
/// \brief Plain-data orbital state: time, position, velocity.
//------------------------------------------------------------------------------
template <std::floating_point Field = double>
struct OrbitalState
{
  using Vector3 = Algebra::Modules::Vectors::Vector3<Field>;

  Field   t;  ///< Current time (s)
  Vector3 r;  ///< Position vector (m)
  Vector3 v;  ///< Velocity vector (m/s)
};

//------------------------------------------------------------------------------
/// \brief Numerov orbit propagator.
///
/// \tparam Field  Floating-point scalar type (default double).
//------------------------------------------------------------------------------
template <std::floating_point Field = double>
class NumerovOrbit
{
  public:

    using Vector3       = Algebra::Modules::Vectors::Vector3<Field>;
    using NumerovState = Numerical::ODE::StormerMethods::NumerovState<Vector3>;

    //--------------------------------------------------------------------------
    /// \brief Bridges TotalAcceleration's (AccelerationInputs→Vector3) call
    ///   signature to NumerovStep's autonomous (Field t, const Vector3&)
    ///   signature.  Time is ignored since gravity is time-invariant.
    //--------------------------------------------------------------------------
    struct AccelerationAdapter
    {
      TwoBodyAcceleration::TotalAcceleration<Field> total_;

      explicit AccelerationAdapter(
        TwoBodyAcceleration::TotalAcceleration<Field>&& total):
        total_{std::move(total)}
      {}

      Vector3 operator()(Field /*t*/, const Vector3& r) const
      {
        return total_(TwoBodyAcceleration::AccelerationInputs<Field>{r});
      }
    };

    using Stepper =
      Numerical::ODE::StormerMethods::NumerovStep<AccelerationAdapter, Field>;

    //--------------------------------------------------------------------------
    /// \brief Construct and run the startup procedure.
    ///
    /// \param accel             Acceleration model (NewtonsGravitation + any
    ///                          perturbations packed into TotalAcceleration).
    /// \param s0                Initial state at t = s0.t.
    /// \param h                 Fixed step size (s).
    /// \param startup_substeps  StormerStep sub-steps for the y_1 bootstrap.
    ///                          Default 128 gives O(h/128)^2 startup error.
    //--------------------------------------------------------------------------
    NumerovOrbit(
      TwoBodyAcceleration::TotalAcceleration<Field> accel,
      const OrbitalState<Field>&                   s0,
      Field                                        h,
      std::size_t startup_substeps = 128):
      h_{h},
      stepper_{AccelerationAdapter{std::move(accel)}},
      t_{s0.t + h},
      state_{stepper_.startup(s0.t, s0.r, s0.v, h, startup_substeps)}
    {}

    //--------------------------------------------------------------------------
    /// \brief Advance one fixed Numerov step (PECECE, 3 force evaluations).
    //--------------------------------------------------------------------------
    void step()
    {
      // Evaluate f_n = f(t_n, y_n) — required by NumerovStep::step().
      // compute_acceleration() delegates to the stored AccelerationAdapter.
      const Vector3 f_n {stepper_.compute_acceleration(t_, state_.position)};
      state_ = stepper_.step(t_, state_, f_n, h_);
      t_ += h_;
    }

    //--------------------------------------------------------------------------
    /// \brief Return the current orbital state (time, position, velocity).
    //--------------------------------------------------------------------------
    OrbitalState<Field> get_state() const
    {
      return OrbitalState<Field>{t_, state_.position, state_.velocity};
    }

    //--------------------------------------------------------------------------
    /// \brief Current time (convenience accessor).
    //--------------------------------------------------------------------------
    Field time() const noexcept { return t_; }

  private:

    ///< Fixed step size
    Field h_;
    ///< NumerovStep holding a copy of AccelerationAdapter
    Stepper stepper_; 
    ///< Current time (initialised to s0.t + h after startup)
    Field t_;
    ///< Numerov 3-point state (y_n, v_n, y_{n-1}, f_{n-1})
    NumerovState state_;
};

} // namespace Propagators
} // namespace Astrodynamics

#endif // ASTRODYNAMICS_PROPAGATORS_NUMEROV_ORBIT_H
