#ifndef NUMERICAL_ODE_STORMER_METHODS_STORMER_STEP_H
#define NUMERICAL_ODE_STORMER_METHODS_STORMER_STEP_H

#include "StormerState.h"

#include <cstddef>
#include <utility> // std::forward

namespace Numerical
{
namespace ODE
{
namespace StormerMethods
{

//------------------------------------------------------------------------------
/// \brief Störmer's modified midpoint rule for y'' = f(x, y).
///
/// \details Implements the leapfrog / Störmer-Verlet sub-stepping algorithm:
///
///   Initialize (half-kick):
///     v_{1/2} = v_0 + (h_eff/2) * f(x_0, y_0)
///
///   Loop m = 1 ... n_eff:
///     y_m     = y_{m-1} + h_eff * v_{m-1/2}          (drift)
///     f_m     = f(x_0 + m * h_eff, y_m)              (force eval)
///     v_{m+1/2} = v_{m-1/2} + h_eff * f_m            (full kick, m < n_eff)
///
///   Final (half-kick back to integer time):
///     z = v_{n_eff - 1/2} + (h_eff/2) * f(x_0 + H, y_{n_eff})
///
///   where h_eff = H / n_eff.
///
///   The error series in h_eff contains only even powers (Gragg 1965), enabling
///   Richardson / Bulirsch-Stoer extrapolation.
///
///   NO base class or inheritance is used. The algorithm is a standalone,
///   composable unit.
///
/// \tparam DerivativeType - callable with signature
///     ContainerT operator()(Field x, const ContainerT& y)
///   returning the acceleration f(x, y). The functor MUST NOT depend on y'.
///
/// \tparam Field - scalar type (default double).
//------------------------------------------------------------------------------
template <typename DerivativeType, typename Field = double>
class StormerStep
{
  public:

    StormerStep() = delete;

    //--------------------------------------------------------------------------
    /// \brief Construct with an rvalue derivative functor (e.g. lambda or
    ///   temporary).
    //--------------------------------------------------------------------------
    explicit StormerStep(DerivativeType&& derivative):
      derivative_{std::forward<DerivativeType>(derivative)}
    {}

    //--------------------------------------------------------------------------
    /// \brief Construct with an lvalue derivative functor.
    //--------------------------------------------------------------------------
    explicit StormerStep(DerivativeType& derivative):
      derivative_{std::forward<DerivativeType>(derivative)}
    {}

    virtual ~StormerStep() = default;

    //--------------------------------------------------------------------------
    /// \brief Execute n_eff leapfrog sub-steps over the interval [x_0, x_0 + H].
    ///
    /// \param x_0      Starting independent variable value.
    /// \param state    Current (position, velocity) pair at x_0.
    /// \param f_0      f(x_0, state.position) — must be pre-computed by caller.
    ///                 This avoids one redundant evaluation (caller may reuse the
    ///                 derivative from a previous step, i.e. FSAL-like behaviour).
    /// \param H        Total interval to advance.
    /// \param n_eff    Number of leapfrog sub-steps. h_eff = H / n_eff.
    ///                 For Bulirsch-Stoer use, the NR3 convention is
    ///                   n_eff = nstep / 2, h_eff = 2 * H / nstep,
    ///                 which halves the derivative-evaluation count versus the
    ///                 first-order midpoint rule.
    ///
    /// \return StormerState with (position, velocity) at x_0 + H, each
    ///   accurate to O(h_eff^2) with only even powers of h_eff in the error.
    //--------------------------------------------------------------------------
    template <typename ContainerT>
    StormerState<ContainerT> step(
      const Field x_0,
      const StormerState<ContainerT>& state,
      const ContainerT& f_0,
      const Field H,
      const std::size_t n_eff) const
    {
      const Field h_eff {H / static_cast<Field>(n_eff)};
      const Field half_h {h_eff / Field{2}};

      // Half-kick: bring velocity from integer time 0 to half-integer time 1/2.
      // v_{1/2} = v_0 + (h_eff/2) * f_0
      ContainerT v_half {state.velocity + half_h * f_0};

      // Position starts at y_0.
      ContainerT y {state.position};

      // Drift-kick loop for m = 1 ... n_eff.
      Field x {x_0};
      ContainerT f_m {f_0};

      for (std::size_t m {1}; m <= n_eff; ++m)
      {
        // Drift: y_m = y_{m-1} + h_eff * v_{m-1/2}
        y += h_eff * v_half;
        x += h_eff;

        // Evaluate force at new position.
        f_m = derivative_(x, y);

        // Full kick (skip on final iteration — velocity remains at half-integer
        // time n_eff - 1/2; the final half-kick is applied after the loop).
        if (m < n_eff)
        {
          v_half += h_eff * f_m;
        }
      }

      // Final half-kick: bring velocity from half-integer time back to integer.
      // z = v_{n_eff - 1/2} + (h_eff/2) * f(x_0 + H, y_{n_eff})
      ContainerT v_out {v_half + half_h * f_m};

      return StormerState<ContainerT>{y, v_out};
    }

    //--------------------------------------------------------------------------
    /// \brief Compute f(x, y) using the stored derivative functor.
    ///
    /// \details Exposed so callers can compute f_0 before calling step(), and
    ///   so they can reuse the final f from a previous call (FSAL pattern).
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

#endif // NUMERICAL_ODE_STORMER_METHODS_STORMER_STEP_H
