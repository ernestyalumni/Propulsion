#ifndef NUMERICAL_ODE_STORMER_METHODS_NUMEROV_STEP_H
#define NUMERICAL_ODE_STORMER_METHODS_NUMEROV_STEP_H

#include "NumerovState.h"
#include "StormerState.h"
#include "StormerStep.h"

#include <cstddef>
#include <utility> // std::forward

namespace Numerical
{
namespace ODE
{
namespace StormerMethods
{

//------------------------------------------------------------------------------
/// \brief Numerov's method for y'' = f(x, y) — order 4 on the k=2 stencil.
///
/// \details Implements the implicit three-point formula (HNW Eq. 10.12, p. 464;
///   B. Numerov 1924):
///
///   y_{n+1} - 2*y_n + y_{n-1} = h^2/12 * (f_{n+1} + 10*f_n + f_{n-1})
///
/// This is the HIGHEST-ORDER STABLE method on the two-step stencil
/// (HNW Theorem 10.4: p <= k+2 for k even; here k=2, p_max = 4).
///
/// Order proof: the generating-polynomial residual
///   rho(e^h) - h^2 * sigma(e^h) = -h^6/240 + O(h^8) = O(h^6)
/// implies order p = 4 (StormerRule.tex §6, HNW Eq. 10.23, Def. 10.2, p. 467).
///
/// Compared with the explicit Störmer (order 2, 1 eval/step):
///   - Position accuracy: O(h^4) vs O(h^2) — two orders better
///   - Velocity accuracy: O(h^2) — same (leapfrog formula)
///   - Cost: 3 force evaluations/step (PECECE) vs 1
///   - Halving h reduces error 16x vs 4x
///
/// SOLVER: PECECE (Predict-Evaluate-Correct-Evaluate-Correct-Evaluate)
/// -----------------------------------------------------------------------
/// The method is implicit (f_{n+1} depends on y_{n+1}). For nonlinear f,
/// we iterate the corrector twice, which recovers full order-4 accuracy:
///
///   P:  y^P    = 2*y_n - y_{n-1} + h^2 * f_n          [explicit Störmer]
///   E:  f^P    = f(x_{n+1}, y^P)
///   C:  y^(1)  = 2*y_n - y_{n-1} + h^2/12*(f^P + 10*f_n + f_{n-1})
///   E:  f^(1)  = f(x_{n+1}, y^(1))
///   C:  y_{n+1}= 2*y_n - y_{n-1} + h^2/12*(f^(1) + 10*f_n + f_{n-1})
///   E:  f_{n+1}= f(x_{n+1}, y_{n+1})   [stored for next step]
///
/// Order analysis of PECECE:
///   Predictor error: O(h^2) global → contamination of f^P: O(h^2).
///   First corrector residual: h^2/12 * O(h^2) = O(h^4) per step → O(h^3) global.
///   Second corrector reduces contamination to O(h^6) per step → O(h^5) global,
///   below the corrector's own leading error O(h^6)/step → O(h^4) global. ✓
///
/// VELOCITY:
///   v_{n+1} = (y_{n+1} - y_n)/h + h/2 * f_{n+1}    [leapfrog, O(h^2)]
///   Positions are O(h^4); velocities are O(h^2).
///   This is a known limitation of pure Numerov integration.
///
/// STARTING PROCEDURE:
///   NumerovStep is a two-step method: requires y_0 AND y_1.
///   startup() uses StormerStep with n_sub sub-steps to produce y_1 with
///   O(h^2 / n_sub^2) error. For full order-4 global convergence, n_sub
///   must be large enough that the startup error is O(h^5). In practice,
///   use n_sub >= 64 or a dedicated 4th-order Runge-Kutta-Nyström starter.
///   HNW requirement (Theorem 10.6, p. 468): |y(x_1) - y_1| = O(h^5).
///
/// \tparam DerivativeType - callable: ContainerT operator()(Field x, const ContainerT& y)
///   Returns the acceleration f(x, y). MUST NOT depend on y'.
///
/// \tparam Field - scalar type (default double).
//------------------------------------------------------------------------------
template <typename DerivativeType, typename Field = double>
class NumerovStep
{
  public:

    NumerovStep() = delete;

    explicit NumerovStep(DerivativeType&& derivative):
      derivative_{std::forward<DerivativeType>(derivative)}
    {}

    explicit NumerovStep(DerivativeType& derivative):
      derivative_{std::forward<DerivativeType>(derivative)}
    {}

    virtual ~NumerovStep() = default;

    //--------------------------------------------------------------------------
    /// \brief Advance one step from x_n to x_{n+1} = x_n + h using PECECE.
    ///
    /// \param x_n    Current independent variable (x at step n).
    /// \param state  NumerovState at step n: position y_n, velocity v_n,
    ///               previous position y_{n-1}, previous force f_{n-1}.
    /// \param f_n    f(x_n, y_n) — must be pre-computed by caller.
    ///               This avoids redundant re-evaluation since f_n from step n
    ///               becomes the stored value for the next call (FSAL-like).
    /// \param h      Step size (fixed; Numerov is not self-starting at variable h).
    ///
    /// \return NumerovState at step n+1:
    ///   - position:  y_{n+1}  (order 4)
    ///   - velocity:  v_{n+1}  (order 2, leapfrog formula)
    ///   - y_prev:    y_n      (becomes y_{n-1} for the next step)
    ///   - f_prev:    f_n      (becomes f_{n-1} for the next step)
    //--------------------------------------------------------------------------
    template <typename ContainerT>
    NumerovState<ContainerT> step(
      const Field           x_n,
      const NumerovState<ContainerT>& state,
      const ContainerT&     f_n,
      const Field           h) const
    {
      const Field h2      {h * h};
      const Field h2_12   {h2 / Field{12}};
      const Field x_next  {x_n + h};

      // RHS constant = 2*y_n - y_{n-1} + h^2/12*(10*f_n + f_{n-1})
      // (the part that doesn't change between iterations)
      const ContainerT rhs_const {
        Field{2} * state.position
        - state.y_prev
        + h2_12 * (Field{10} * f_n + state.f_prev)
      };

      // --- PREDICT (P): explicit Störmer ---
      // y^P = 2*y_n - y_{n-1} + h^2 * f_n
      ContainerT y_pred {
        Field{2} * state.position - state.y_prev + h2 * f_n
      };

      // --- EVALUATE (E) ---
      ContainerT f_iter {derivative_(x_next, y_pred)};

      // --- CORRECT (C) first pass ---
      // y^(1) = rhs_const + h^2/12 * f^P
      ContainerT y_corr {rhs_const + h2_12 * f_iter};

      // --- EVALUATE (E) second pass ---
      f_iter = derivative_(x_next, y_corr);

      // --- CORRECT (C) second pass ---
      // y_{n+1} = rhs_const + h^2/12 * f^(1)
      ContainerT y_new {rhs_const + h2_12 * f_iter};

      // --- EVALUATE (E) final: store f_{n+1} for next step ---
      ContainerT f_new {derivative_(x_next, y_new)};

      // --- VELOCITY at n+1 via leapfrog (order 2) ---
      // v_{n+1} = (y_{n+1} - y_n)/h + h/2 * f_{n+1}
      ContainerT v_new {
        (y_new - state.position) / h + (h / Field{2}) * f_new
      };

      // Slide history window: new state's y_prev = y_n, f_prev = f_n
      return NumerovState<ContainerT>{
        std::move(y_new),
        std::move(v_new),
        state.position,   // y_{n-1} for next step = current y_n
        f_n               // f_{n-1} for next step = current f_n
      };
    }

    //--------------------------------------------------------------------------
    /// \brief Build a NumerovState at step 1 from initial conditions (y_0, v_0).
    ///
    /// \details Uses StormerStep with n_sub sub-steps over one full step h to
    ///   produce y_1.  For full order-4 convergence of subsequent Numerov steps,
    ///   HNW Theorem 10.6 requires |y(x_1) - y_1| = O(h^5).  StormerStep gives
    ///   O(h^2 / n_sub^2) positional error; choose n_sub large enough that this
    ///   is below O(h^5), i.e. n_sub >> h^{-3/2} (roughly n_sub >= 64 for
    ///   moderate h values in orbital mechanics).
    ///
    ///   In production, replace this with a single 4th-order Runge-Kutta-Nyström
    ///   step for an O(h^5) startup in a fixed number of evaluations.
    ///
    /// \param x_0   Starting independent variable.
    /// \param y_0   Initial position.
    /// \param v_0   Initial velocity.
    /// \param h     Step size (same h that will be used for subsequent steps).
    /// \param n_sub Number of Störmer-Verlet sub-steps for the startup step.
    ///              Default 128; increase for tighter startup accuracy.
    ///
    /// \return NumerovState ready for use as input to step().
    //--------------------------------------------------------------------------
    template <typename ContainerT>
    NumerovState<ContainerT> startup(
      const Field       x_0,
      const ContainerT& y_0,
      const ContainerT& v_0,
      const Field       h,
      const std::size_t n_sub = 128) const
    {
      // Evaluate f_0 = f(x_0, y_0)
      ContainerT f_0 {derivative_(x_0, y_0)};

      // One StormerStep of size h to get (y_1, v_1)
      StormerState<ContainerT> s0 {y_0, v_0};
      StormerStep stormer_start {derivative_};
      StormerState<ContainerT> s1 {
        stormer_start.step(x_0, s0, f_0, h, n_sub)
      };

      // f_{-1} (i.e., f(x_0, y_0)) will be f_prev for the first Numerov step.
      // f_0 serves as f_n for the first call to step().
      // So we return state at index n=1:
      //   position = y_1, velocity = v_1
      //   y_prev   = y_0
      //   f_prev   = f_0
      ContainerT f_1 {derivative_(x_0 + h, s1.position)};

      // Velocity at step 1 via leapfrog: consistent with step() convention.
      ContainerT v_1 {
        (s1.position - y_0) / h + (h / Field{2}) * f_1
      };

      return NumerovState<ContainerT>{
        s1.position,   // y_1
        v_1,           // v_1 (leapfrog, O(h^2))
        y_0,           // y_prev = y_0
        f_0            // f_prev = f_0 = f(x_0, y_0)
      };
    }

    //--------------------------------------------------------------------------
    /// \brief Evaluate f(x, y) via the stored derivative functor.
    ///
    /// \details Exposed so callers can compute f_n before calling step().
    //--------------------------------------------------------------------------
    template <typename ContainerT>
    ContainerT compute_acceleration(const Field x, const ContainerT& y) const
    {
      return derivative_(x, y);
    }

  private:

    DerivativeType derivative_;
};

} // namespace StormerMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_STORMER_METHODS_NUMEROV_STEP_H
