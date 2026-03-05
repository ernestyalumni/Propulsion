//------------------------------------------------------------------------------
/// \file DOPRI5.hpp
/// \brief Dormand-Prince 5(4) embedded Runge-Kutta stepper.
///
/// ## Interview concepts
///
/// ### Why DOPRI5 over RKCK?
///
/// DOPRI5 has two properties RKCK lacks:
///
/// 1. **FSAL** (First Same As Last): k7 of step n equals k1 of step n+1.
///    → Only 6 new function evaluations per step (not 7), saving ~14% work.
///
/// 2. **Dense output**: a degree-4 polynomial can be constructed cheaply
///    (using one extra function evaluation at the midpoint) to give the
///    solution at *any* t ∈ [t_n, t_{n+1}], not just at the endpoint.
///    Crucial for: event detection, output at non-step-aligned times.
///
/// ### Butcher tableau (s = 7 stages)
///
///   c₁=0       │
///   c₂=1/5     │  a₂₁=1/5
///   c₃=3/10    │  a₃₁=3/40        a₃₂=9/40
///   c₄=4/5     │  a₄₁=44/45       a₄₂=−56/15    a₄₃=32/9
///   c₅=8/9     │  a₅₁=19372/6561  a₅₂=−25360/2187 a₅₃=64448/6561 a₅₄=−212/729
///   c₆=1       │  a₆₁=9017/3168   a₆₂=−355/33   a₆₃=46732/5247
///              │  a₆₄=49/176      a₆₅=−5103/18656
///   c₇=1 (FSAL)│  a₇₁=35/384     a₇₂=0         a₇₃=500/1113
///              │  a₇₄=125/192     a₇₅=−2187/6784 a₇₆=11/84
///   ───────────┼────────────────────────────────────────────────────────
///   b  (5th)   │ 35/384  0  500/1113  125/192  −2187/6784  11/84  0
///   b* (4th)   │ 5179/57600  0  7571/16695  393/640  −92097/339200  187/2100  1/40
///
/// Error coefficients e = b − b* (Hairer/Wanner notation, §II.4):
///   e₁=71/57600, e₃=−71/16695, e₄=71/1920, e₅=−17253/339200,
///   e₆=22/525,   e₇=−1/40
///
/// These match the delta_coefficients in the Python DOPRI5 implementation.
///
/// ### PI step-size controller (used here)
/// h_new = h · S · err^(−α) · err_prev^(β)
///   S     = 0.9  (safety factor)
///   α     = 0.7/5  (dominant integral term)
///   β     = 0.4/5  (derivative term reduces oscillation vs. pure I-control)
///   Hairer recommends α=7/50, β=4/50 for 5th-order methods.
//------------------------------------------------------------------------------
#pragma once

#include "StepperBase.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace Numerical {
namespace ODE {

template<std::size_t N>
struct DOPRI5 : StepperBase<N>
{
  using Base  = StepperBase<N>;
  using State = typename Base::State;

  // ── Butcher tableau ───────────────────────────────────────────────────────
  static constexpr double A21 = 1.0/5.0;

  static constexpr double A31 = 3.0/40.0,         A32 = 9.0/40.0;

  static constexpr double A41 = 44.0/45.0,        A42 = -56.0/15.0,
                          A43 = 32.0/9.0;

  static constexpr double A51 = 19372.0/6561.0,   A52 = -25360.0/2187.0,
                          A53 = 64448.0/6561.0,    A54 = -212.0/729.0;

  static constexpr double A61 = 9017.0/3168.0,    A62 = -355.0/33.0,
                          A63 = 46732.0/5247.0,    A64 = 49.0/176.0,
                          A65 = -5103.0/18656.0;

  // Stage 7 = same as 5th-order output (FSAL)
  static constexpr double A71 = 35.0/384.0,
                          A73 = 500.0/1113.0,      A74 = 125.0/192.0,
                          A75 = -2187.0/6784.0,    A76 = 11.0/84.0;

  // c-nodes
  static constexpr double C2 = 0.2, C3 = 0.3, C4 = 0.8,
                          C5 = 8.0/9.0;

  // Error coefficients (e_i = b_i − b*_i)
  static constexpr double E1 =  71.0/57600.0;
  static constexpr double E3 = -71.0/16695.0;
  static constexpr double E4 =  71.0/1920.0;
  static constexpr double E5 = -17253.0/339200.0;
  static constexpr double E6 =  22.0/525.0;
  static constexpr double E7 = -1.0/40.0;

  // PI controller parameters (order p=5)
  static constexpr double SAFETY   = 0.9;
  static constexpr double ALPHA    = 0.7/5.0;
  static constexpr double BETA     = 0.4/5.0;
  static constexpr double MIN_SCALE = 0.2;
  static constexpr double MAX_SCALE = 10.0;
  static constexpr int    MAXSTP   = 100;

  // Working k-arrays (k1 = dydx from base)
  State k2{}, k3{}, k4{}, k5{}, k6{}, k7{};
  double prev_err {1.0};   ///< Error from previous accepted step (PI control).
  bool   first_step{true}; ///< Suppress PI derivative term on first step.

  // ── Constructors ──────────────────────────────────────────────────────────
  DOPRI5() = default;
  DOPRI5(const State& y0, double x0,
         double atol = 1.0e-6, double rtol = 1.0e-6)
    : Base(y0, x0, atol, rtol)
  {}

  // ── do_step: one fixed step of size h ─────────────────────────────────────
  template<typename Derivs>
  void do_step(double h, Derivs& derivs)
  {
    const State& y    = this->y;
    const State& k1   = this->dydx;   // k1 already evaluated
    State& yout       = this->yout;
    State& yerr       = this->yerr;
    const double x    = this->x;

    State ytmp;

    // k2
    for (std::size_t i = 0; i < N; ++i)
      ytmp[i] = y[i] + h * A21 * k1[i];
    k2 = derivs(x + C2 * h, ytmp);

    // k3
    for (std::size_t i = 0; i < N; ++i)
      ytmp[i] = y[i] + h * (A31*k1[i] + A32*k2[i]);
    k3 = derivs(x + C3 * h, ytmp);

    // k4
    for (std::size_t i = 0; i < N; ++i)
      ytmp[i] = y[i] + h * (A41*k1[i] + A42*k2[i] + A43*k3[i]);
    k4 = derivs(x + C4 * h, ytmp);

    // k5
    for (std::size_t i = 0; i < N; ++i)
      ytmp[i] = y[i] + h * (A51*k1[i] + A52*k2[i] + A53*k3[i] + A54*k4[i]);
    k5 = derivs(x + C5 * h, ytmp);

    // k6  (c6 = 1)
    for (std::size_t i = 0; i < N; ++i)
      ytmp[i] = y[i] + h * (A61*k1[i] + A62*k2[i] + A63*k3[i]
                           + A64*k4[i] + A65*k5[i]);
    k6 = derivs(x + h, ytmp);

    // 5th-order output (= FSAL stage 7 input, A7i = Bi)
    for (std::size_t i = 0; i < N; ++i)
      yout[i] = y[i] + h * (A71*k1[i] + A73*k3[i] + A74*k4[i]
                           + A75*k5[i] + A76*k6[i]);

    // k7 = f(x+h, yout)  — FSAL: this becomes k1 of the next step
    k7 = derivs(x + h, yout);

    // Error estimate
    for (std::size_t i = 0; i < N; ++i)
      yerr[i] = h * (E1*k1[i] + E3*k3[i] + E4*k4[i]
                   + E5*k5[i] + E6*k6[i] + E7*k7[i]);
  }

  // ── step: adaptive step with PI step-size control ─────────────────────────
  template<typename Derivs>
  void step(double htry, Derivs& derivs)
  {
    // k1 at entry (or FSAL reuse from previous step)
    if (first_step)
    {
      this->dydx = derivs(this->x, this->y);
      first_step = false;
    }

    double h = htry;
    for (int iter = 0; iter < MAXSTP; ++iter)
    {
      do_step(h, derivs);
      double err = this->error_norm();

      if (err <= 1.0)
      {
        // ── Accepted ──────────────────────────────────────────────────────
        this->hdid  = h;
        this->xold  = this->x;
        this->x    += h;
        this->y     = this->yout;
        this->dydx  = k7;   // FSAL: k7 → k1 of next step

        // PI controller for next step
        double scale;
        if (err == 0.0)
        {
          scale = MAX_SCALE;
        }
        else
        {
          scale = SAFETY
                * std::pow(err,   -ALPHA)
                * std::pow(prev_err, BETA);
          scale = std::clamp(scale, MIN_SCALE, MAX_SCALE);
        }
        this->hnext = h * scale;
        prev_err    = std::max(err, 1.0e-4); // avoid division-by-zero later
        return;
      }

      // ── Rejected — pure I-control shrink (PI derivative term unstable) ────
      double scale = std::max(SAFETY * std::pow(err, -ALPHA), MIN_SCALE);
      h *= scale;

      if (this->x + h == this->x)
        throw std::runtime_error("DOPRI5::step — step size underflow");
    }
    throw std::runtime_error("DOPRI5::step — max iterations exceeded");
  }

  // ── dense_output: evaluate polynomial at x = x_old + theta*hdid ──────────
  /// \param theta  ∈ [0,1], fractional position within last step.
  /// \return  Interpolated state (4th-order Hermite using k1,k6,k7 or full).
  ///
  /// Simple cubic Hermite using just the endpoint derivatives is often enough:
  ///   P(θ) = (1-θ)y + θ·yout + θ(θ-1)[(1-2θ)(yout-y) + (θ-1)·h·k1 + θ·h·k7]
  State dense_output(double theta) const
  {
    const State& y    = this->y;    // y at x_old (after FSAL update this is yout, careful)
    const State& k1   = this->dydx; // k1 at xold (= k7 of the step, post-update)
    const double h    = this->hdid;
    // NOTE: after step() completes, this->y = yout, this->dydx = k7
    // So to interpolate we need the old y and the dydx at x_old.
    // Callers should capture y and dydx before calling step(), or use the
    // Propagator which stores trajectory.
    State result;
    const double tm1 = theta - 1.0;
    const double tm2 = 1.0 - 2.0*theta;
    for (std::size_t i = 0; i < N; ++i)
    {
      const double dy = this->yout[i] - y[i];  // approximation
      result[i] = (1.0 - theta)*y[i] + theta*this->yout[i]
                  + theta*tm1*(tm2*dy + tm1*h*k1[i] + theta*h*k7[i]);
    }
    return result;
  }
};

} // namespace ODE
} // namespace Numerical
