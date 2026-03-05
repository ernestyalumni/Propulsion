//------------------------------------------------------------------------------
/// \file RKCK.hpp
/// \brief Cash-Karp Runge-Kutta 4(5) embedded stepper.
///
/// ## Interview concepts
///
/// ### Why Cash-Karp (RKCK)?
/// It is the embedded pair used in NR §17.2 ("rkck" / "rkqs").
/// One step computes *two* RK approximations of different orders:
///   y_5  – 5th-order accurate  (used to advance the solution)
///   y_4  – 4th-order accurate  (embedded; only coefficients differ)
///   yerr = y_5 - y_4            (local error estimate; costs *no extra* f evals)
///
/// ### Butcher tableau (s = 6 stages)
///
///   c₁=0       │
///   c₂=1/5     │  a₂₁=1/5
///   c₃=3/10    │  a₃₁=3/40       a₃₂=9/40
///   c₄=3/5     │  a₄₁=3/10       a₄₂=−9/10     a₄₃=6/5
///   c₅=1       │  a₅₁=−11/54     a₅₂=5/2       a₅₃=−70/27    a₅₄=35/27
///   c₆=7/8     │  a₆₁=1631/55296 a₆₂=175/512   a₆₃=575/13824
///              │  a₆₄=44275/110592 a₆₅=253/4096
///   ───────────┼───────────────────────────────────────────────────
///   b  (5th)   │  37/378     0    250/621    125/594      0      512/1771
///   b* (4th)   │  2825/27648 0    18575/48384 13525/55296 277/14336  1/4
///
/// e_i = b_i − b*_i  (error coefficients; used to form yerr)
///
/// ### Adaptive step-size control (NR §17.2.1)
/// After computing yerr, the scaled RMS norm (err) is evaluated.
/// If err > 1:  rejected → h shrunk by factor S·err^(−1/4), retry.
/// If err ≤ 1:  accepted → h grown for next step by S·err^(−1/5),
///              capped at 5× growth, floored at 0.1× reduction.
///
/// The exponent −1/(p+1) for shrink and −1/p for growth come from
/// setting the leading-error term equal to the tolerance.
//------------------------------------------------------------------------------
#pragma once

#include "StepperBase.hpp"
#include <array>
#include <cmath>
#include <stdexcept>

namespace Numerical {
namespace ODE {

template<std::size_t N>
struct RKCK : StepperBase<N>
{
  using Base  = StepperBase<N>;
  using State = typename Base::State;

  // ── Butcher tableau constants (static, shared by all instances) ───────────
  static constexpr double A21 = 1.0/5.0;

  static constexpr double A31 = 3.0/40.0,      A32 = 9.0/40.0;

  static constexpr double A41 = 3.0/10.0,      A42 = -9.0/10.0,
                          A43 = 6.0/5.0;

  static constexpr double A51 = -11.0/54.0,    A52 = 5.0/2.0,
                          A53 = -70.0/27.0,     A54 = 35.0/27.0;

  static constexpr double A61 = 1631.0/55296.0, A62 = 175.0/512.0,
                          A63 = 575.0/13824.0,  A64 = 44275.0/110592.0,
                          A65 = 253.0/4096.0;

  // 5th-order weights (advance)
  static constexpr double B1 = 37.0/378.0,    B3 = 250.0/621.0,
                          B4 = 125.0/594.0,   B6 = 512.0/1771.0;

  // Error = b5 − b4*
  static constexpr double E1 =  37.0/378.0  - 2825.0/27648.0;
  static constexpr double E3 = 250.0/621.0  - 18575.0/48384.0;
  static constexpr double E4 = 125.0/594.0  - 13525.0/55296.0;
  static constexpr double E5 =              - 277.0/14336.0;
  static constexpr double E6 = 512.0/1771.0 - 1.0/4.0;

  // c-nodes (time fractions) for stages
  static constexpr double C2 = 0.2, C3 = 0.3, C4 = 0.6, C6 = 7.0/8.0;

  // Step-size control parameters
  static constexpr double SAFETY  = 0.9;
  static constexpr double PGROW   = -0.2;   // exponent for growth  (−1/5)
  static constexpr double PSHRINK = -0.25;  // exponent for shrink  (−1/4)
  static constexpr double ERRCON  = 1.89e-4; // = (5/SAFETY)^(1/PGROW)
  static constexpr int    MAXSTP  = 50;

  // Working k-arrays (avoid alloc per step)
  State k2{}, k3{}, k4{}, k5{}, k6{};

  // ── Constructors ──────────────────────────────────────────────────────────
  RKCK() = default;
  RKCK(const State& y0, double x0,
       double atol = 1.0e-6, double rtol = 1.0e-6)
    : Base(y0, x0, atol, rtol)
  {}

  // ── do_step: one fixed step of size h ─────────────────────────────────────
  /// \param h        Step size.
  /// \param derivs   Callable: State derivs(double t, const State& y).
  template<typename Derivs>
  void do_step(double h, Derivs& derivs)
  {
    State& y    = this->y;
    State& dydx = this->dydx;
    State& yout = this->yout;
    State& yerr = this->yerr;
    double x    = this->x;

    // k1 = dydx (already evaluated at entry point)

    // k2
    State ytmp;
    for (std::size_t i = 0; i < N; ++i)
      ytmp[i] = y[i] + h * A21 * dydx[i];
    k2 = derivs(x + C2 * h, ytmp);

    // k3
    for (std::size_t i = 0; i < N; ++i)
      ytmp[i] = y[i] + h * (A31 * dydx[i] + A32 * k2[i]);
    k3 = derivs(x + C3 * h, ytmp);

    // k4
    for (std::size_t i = 0; i < N; ++i)
      ytmp[i] = y[i] + h * (A41 * dydx[i] + A42 * k2[i] + A43 * k3[i]);
    k4 = derivs(x + C4 * h, ytmp);

    // k5
    for (std::size_t i = 0; i < N; ++i)
      ytmp[i] = y[i] + h * (A51*dydx[i] + A52*k2[i] + A53*k3[i] + A54*k4[i]);
    k5 = derivs(x + h, ytmp);  // c5 = 1

    // k6
    for (std::size_t i = 0; i < N; ++i)
      ytmp[i] = y[i] + h * (A61*dydx[i] + A62*k2[i] + A63*k3[i]
                           + A64*k4[i] + A65*k5[i]);
    k6 = derivs(x + C6 * h, ytmp);

    // 5th-order output (B2=0, B5=0)
    for (std::size_t i = 0; i < N; ++i)
      yout[i] = y[i] + h * (B1*dydx[i] + B3*k3[i] + B4*k4[i] + B6*k6[i]);

    // Error estimate  (E2=0, so k2 term absent)
    for (std::size_t i = 0; i < N; ++i)
      yerr[i] = h * (E1*dydx[i] + E3*k3[i] + E4*k4[i] + E5*k5[i] + E6*k6[i]);
  }

  // ── step: adaptive step with error control ────────────────────────────────
  /// Tries htry; shrinks and retries on rejection; sets hdid, hnext.
  template<typename Derivs>
  void step(double htry, Derivs& derivs)
  {
    // Evaluate derivative at entry
    this->dydx = derivs(this->x, this->y);

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
        // dydx at new point needs recompute next call (RKCK is not FSAL)
        // Compute recommended next step
        if (err > ERRCON)
          this->hnext = SAFETY * h * std::pow(err, PGROW);
        else
          this->hnext = 5.0 * h;   // cap growth at 5×
        return;
      }

      // ── Rejected — shrink h ───────────────────────────────────────────────
      double htemp = SAFETY * h * std::pow(err, PSHRINK);
      h = std::max(htemp, 0.1 * h);   // floor: no more than 10× shrink

      if (this->x + h == this->x)
        throw std::runtime_error("RKCK::step — step size underflow");
    }
    throw std::runtime_error("RKCK::step — max iterations exceeded");
  }
};

} // namespace ODE
} // namespace Numerical
