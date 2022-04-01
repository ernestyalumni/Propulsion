#ifndef NUMERICAL_ODE_STEPPER_DOPR5_H
#define NUMERICAL_ODE_STEPPER_DOPR5_H

#include "StepperBase.h"

#include <cmath>
#include <limits>
#include <vector>

namespace Numerical
{
namespace ODE
{

//------------------------------------------------------------------------------
/// \details Was called stepperdopr5.h on pp. 917, Numerical Recipes, 17.2
/// Adaptive Stepsize Control for Runge-Kutta.
//------------------------------------------------------------------------------
template <class D>
struct StepperDopr5 : StepperBase
{
  // Make the type of derivs (derivatives) available to odeint.
  typedef D = DerivativeType;

  std::vector<double> k2_;
  std::vector<double> k3_;
  std::vector<double> k4_;
  std::vector<double> k5_;
  std::vector<double> k6_;

  std::vector<double> rcont1_;
  std::vector<double> rcont2_;
  std::vector<double> rcont3_;
  std::vector<double> rcont4_;
  std::vector<double> rcont5_;

  std::vector<double> dydxnew_;

  StepperDopr5(
    std::vector<double>& yy,
    std::vector<double>& dydxx,
    std::vector<double>& xx,
    const double a_toll,
    const double r_toll,
    bool dense);

  void step(const double htry, D& derivatives);

  double dense_out(const std::size_t i, const double x, const double h);

  double error();

  struct Controller
  {
    double h_next_;
    double err_old_;
    bool reject_;

    Controller();

    bool success(const double err, double& h);
  };

  Controller con_;
};

//------------------------------------------------------------------------------
/// \brief Input to the ctor are the dependent variable y[0..n-1] and its
/// derivative dydx[0..n-1] at the starting value of the independent variable x.
/// Also input are the absolute and relative tolerances, atol and rtol, and the
/// boolean dense, which is true if dense output is required.
//------------------------------------------------------------------------------
template <class D>
StepperDopr5<D>::StepperDopr5(
  std::vector<double>& yy,
  std::vector<double>& dydxx,
  double& xx,
  const double atoll,
  const double rtoll,
  bool dense
  ):
  StepperBase{yy, dydxx, xx, atoll, rtoll, dense},
  k2_(n_),
  k3_(n_),
  k4_(n_),
  k5_(n_),
  k6_(n_),
  rcont1_(n_),
  rcont2_(n_),
  rcont3_(n_),
  rcont4_(n_),
  rcont5_(n_),
  dydxnew_(n_)
{
  EPS_ = std::numeric_limits<double>::epsilon();
}

//------------------------------------------------------------------------------
/// \ref pp. 917, Ch. 17.2 Adaptive Stepsize Control for Runge-Kutta
/// \details The step method is the actual stepper. It attempts a step, invokes
/// the controller to decide whether to accept the step or try again with a
/// smaller stepsize, and sets up the coefficients in case dense output is
/// needed between x and x + h.
///
/// \brief Attempts a step with stepsize htry. On output, y and x are replaced
/// by their new values, hdid is the stepsize that was actually accomplished,
/// and hnext is the estimated next stepsize.
//------------------------------------------------------------------------------
template <class D>
void StepperDopr5<D>::step(const double htry, D& derivatives)
{
  // Step stepsize to the initial trial value.
  double h {htry};

  for (;;)
  {
    // Take a step.
    dy(h, derivatives);
    // Evaluate accuracy.
    double err {error()};

    // Step rejected. Try again with reduced h set by controller.
    //if (con_.success(err, h))
    //{
    //  break;
    //}
    if (std::abs(h) <= std::abs(x_) * EPS_)
    {
      throw("stepsize underflow in StepperDopr5");
    }
  }

  // Step succeeded. Compute coefficients for dense output.
  if (dense_)
  {
    //prepare_dense(h, derivatives);
  }

  dydx_ = dydxnew_;
  y_ = y_out_;
  // Used for dense output.
  x_old_ = x_;
  x_ += (h_did_ = h);
  //h_next_ = con_.h_next_;
}

//------------------------------------------------------------------------------
/// \details Evaluate interpolating polynomial for y[i] at location x, where
/// xold <= x <= xold + h
//------------------------------------------------------------------------------
template <class D>
double StepperDopr5<D>::dense_out(
  const std::size_t i,
  const double x,
  const double h)
{
  const double s {(x - x_old_) / h};
  const double s1 {1.0 - s};
  return rcont1_[i] + 
    s * (rcont2_[i] + s1 * (rcont3_[i] + s * (rcont4_[i] + s1 * rcont5_[i])));
}

//------------------------------------------------------------------------------
/// \details Use yerr to compute norm of scaled error estimate. A value less
/// than one means the step was successful.
//------------------------------------------------------------------------------
template <class D>
double StepperDopr5<D>::error()
{
  double err {0.0};
  double sk;
  for (int i {0}; i < n; ++i)
  {
    sk = a_tolerance_ + r_tolerance_ * std::max(
      std::abs(y_[i]),
      std::abs(y_out_[i]));
    err += std::pow(y_err_[i] / sk, 2);
  }
  return std::sqrt(err / n_);
}

} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_STEPPER_DOPR5_H
