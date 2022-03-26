#ifndef NUMERICAL_ODE_ODE_INT_H
#define NUMERICAL_ODE_ODE_INT_H

#include <cmath>
#include <vector>

namespace Numerical
{
namespace ODE
{

//------------------------------------------------------------------------------
/// \ref Numerical Recipes, 3rd. Ed. Press, Teukolsky, Vetterling, Flannery.
/// 17.0.2 The Odeint Object. pp. 902
/// \brief Driver for ODE solvers with adaptive stepsize control. The template
/// parameter should be 1 of the derived classes of StepperBase defining a
/// particular integration algorithm.
//------------------------------------------------------------------------------
template <class Stepper, class Output>
class OdeInt
{
  public:

    // Take at most MAXSTP steps.
    static constexpr std::size_t MAXSTP {50000};

    using Derivative = Stepper::DerivativeType;

    //--------------------------------------------------------------------------
    /// \details Ctor simply initializes a bunch of things, including a call to
    /// stepper ctor.
    ///
    /// Ctor sets everything up. The routine integrates starting values ystart_t
    /// [0..nvar-1] from xx1 to xx2 with absolute tolerance atol and relative
    /// tolerance rtol. The quantity h1 should be set as a guessed first
    /// stepsize, hmin as the minimum allowed stepsize (can be 0). An Output
    /// object out should be input to control the saving of intermediate values.
    /// On output, nok and nbad are number of good and bad (but retried and
    /// fixed) steps taken, and ystart_ is replaced by values at end of
    /// integration interval. derivatives_ is the user-supplied routine
    /// (function or functor) for calculating the right-hand side derivative.
    //--------------------------------------------------------------------------
    OdeInt(
      std::vector<double>& y_start_t,
      const double xx1,
      const double xx2,
      const double a_tolerance,
      const double r_tolerance,
      const double h1,
      const double h_min_n,
      Output& outt,
      Stepper::DerivativeType& derivatives);

    //void integrate();

    double EPS_;
    int nok_;
    int nbad_;
    int n_var_;
    double x1_;
    double x2_;
    double hmin_;

    // True if dense output requested by out.
    bool dense_;

    std::vector<double> y_;
    std::vector<double> dy_dx_;
    std::vector<double>& ystart_;
    Output& output_;

    Derivative& derivatives_;

    Stepper s_;
    int n_stp_;
    double x_;
    double h_;
};

template <class Stepper, class Output>
OdeInt<Stepper, Output>::OdeInt(
  std::vector<double>& y_start_t,
  const double xx1,
  const double xx2,
  const double a_tolerance,
  const double r_tolerance,
  const double h1,
  const double h_min_n,
  Output& outt,
  Stepper::DerivativeType& derivatives
  ):
  n_var_(y_start_t.size()),
  y_{n_var_},
  dy_dx_{n_var_},
  ystart_{y_start_t},
  x_{xx1},
  nok_{0},
  nbad_{0},
  x1_{xx1},
  x2_{xx2},
  hmin_{h_min_n},
  dense_{outt.dense_},
  output_{outt},
  derivatives_{derivatives},
  s_{y_, dy_dx_, x_, a_tolerance, r_tolerance, dense_}
{
  EPS_ = std::numeric_limits<double>::epsilon();
  
  h_ = std::signbit(x2_ - x1_) ? -std::abs(h1) : std::abs(h1);
  //h = SIGN(h1_, x2_ - x1_);
  
  for (int i {0}; i < n_var_; ++i)
  {
    y_[i] = ystart_[i];
  }
  output_.init(s_.n_eqns_, x1_, x2_);
}

/*
template <class Stepper, class Output>
void OdeInt<Stepper, Output>::integrate()
{
  derivatives_(x_, y_, dy_dx_);
  if (dense_)
  {
    output_.out(-1, x_, y_, s_, h_);
  }
  else
  {
    output_.save(x_, y_);
  }

  for (n_stp_ = 0; n_stp_ < MAXSTP; ++n_stp_)
  {
    // If stepsize can oversheet, decrease.
    if ((x_ + h_ * 1.0001 -x2_) * (x2_ - x1_) > 0.0)
    {
      h_ = x2_ - x_;
    }
    // Take a step.
    s_.step(h_, derivatives_);

    if (s_.hdid_ == h_)
    {
      ++nok_;
    }
    else
    {
      ++nbad_;
    }

    if (dense_)
    {
      output_.out(n_stp_, x_, y_, s_, s_.hdid_);
    }
    else
    {
      output_.save(x_, y_);
    }

    if ((x_ - x2_) * (x2_ - x1_) >= 0.0)
    {
      for (int i {0}; i < n_var_; ++i)
      {
        ystart_[i] = y_[i];
      }
      if (
        output_.kmax_ > 0 &&
          std::abs(output_.xsave[output_.count_ - 1] - x2_) >
            100.0 * std::abs(x2_) * EPS_)
      {
        output_.save(x_, y_);
      }
      return;
    }
    if (std::abs(s_.hnext_) <= hmin_)
    {
      throw("Step size too small in OdeInt");
    }
    h_ * s_.hnext_;
  }

  throw("Too many steps in routine OdeInt");
}
*/

} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_ODE_INT_H
