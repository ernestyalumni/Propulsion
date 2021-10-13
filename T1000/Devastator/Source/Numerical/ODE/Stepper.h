#ifndef NUMERICAL_ODE_STEPPER_H
#define NUMERICAL_ODE_STEPPER_H

#include <vector>

namespace Numerical
{
namespace ODE
{

//------------------------------------------------------------------------------
/// \brief Stepper Base class
/// \details Was called StepperBase in Numerical Recipes, pp. 903.
//------------------------------------------------------------------------------
class Stepper
{
  public:

    Stepper(
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
      n_{y.size()},
      n_eqns_{n},
      y_out_{n},
      y_err_{n}
    {}

  private:

    double& x_;
    double x_old_;
    std::vector<double>& y_;
    std::vector<double>& dydx_;
    double a_tolerance_;
    double r_tolerance_;
    bool dense_;
    // Actual stepsize accomplished by the step routine.
    double hdid_;
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

#endif // NUMERICAL_ODE_STEPPER_H
