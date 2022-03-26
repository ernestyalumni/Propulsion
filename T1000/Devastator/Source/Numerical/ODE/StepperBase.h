#ifndef NUMERICAL_ODE_STEPPER_BASE_H
#define NUMERICAL_ODE_STEPPER_BASE_H

#include <vector>

namespace Numerical
{
namespace ODE
{

//------------------------------------------------------------------------------
/// \brief Stepper Base class
/// Base class for all ODE algorithms.
/// \details Was called StepperBase in Numerical Recipes, pp. 903.
//------------------------------------------------------------------------------
class StepperBase
{
  public:

    //--------------------------------------------------------------------------
    /// \details Input to the ctor are dependent variable vector y[0..n-1] and
    /// its derivative dydx[0...n-1] at the starting value of the independent
    /// variable x. Als input are the absolute and relative tolerances, atol and
    /// rtol, and boolean dense, which is true if dense output is required.
    //--------------------------------------------------------------------------
    StepperBase(
      std::vector<double>& yy,
      std::vector<double>& dydxx,
      double& xx,
      const double a_toll,
      const double r_toll,
      bool dens
      ):
      x_{xx},
      y_{yy},
      dydx_{dydxx},
      a_tolerance_{a_toll},
      r_tolerance_{r_toll},
      dense_{dens},
      n_{static_cast<int>(y_.size())},
      n_eqns_{n_},
      // Invoke std::vector constructor constructing with size.
      y_out_(n_),
      y_err_(n_)
    {}

    double& x_;
    double x_old_;
    std::vector<double>& y_;
    std::vector<double>& dydx_;
    double a_tolerance_;
    double r_tolerance_;
    bool dense_;
    // Actual stepsize accomplished by the step routine.
    double hdid_;
    // Stepsize predicted by the controller for the next step.
    double hnext_;
    double EPS_;
    int n_;
    // n_eqns_ = n except for StepperStoerm.
    int n_eqns_;
    std::vector<double> y_out_;
    std::vector<double> y_err_;
};

} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_STEPPER_BASE_H
