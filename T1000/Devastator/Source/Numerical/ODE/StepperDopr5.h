#ifndef NUMERICAL_ODE_STEPPER_DOPR5_H
#define NUMERICAL_ODE_STEPPER_DOPR5_H

#include "StepperBase.h"

#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace Numerical
{
namespace ODE
{

using StdFunctionDerivativeType =
  std::function<void(double, std::vector<double>, std::vector<double>)>;

//------------------------------------------------------------------------------
/// \details Was called stepperdopr5.h on pp. 917, Numerical Recipes, 17.2
/// Adaptive Stepsize Control for Runge-Kutta.
//------------------------------------------------------------------------------
template <class D>
struct StepperDopr5 : StepperBase
{
  // Make the type of derivs (derivatives) available to odeint.
  using DerivativeType = D;

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
    double& xx,
    const double a_toll,
    const double r_toll,
    bool dense);

  void step(const double htry, D& derivatives);

  void dy(const double h, D& derivatives);

  void prepare_dense(const double h, D& derivatives);

  double dense_out(const std::size_t i, const double x, const double h);

  double error();

  struct Controller
  {
    double h_next_;
    double err_old_;
    bool reject_;

    // TODO: define implementation of ctor.
    Controller() = default;

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
/// \details Given values for n variables y[0...n-1] and their derivatives
/// dydx[0..n-1] known at x, use the 5th-order Dormand-Prince Runge-Kutta
/// method to advance solution over interval h and store incremented variables
/// in y_out_[0...n-1]. Also store estimate of local truncation error in y_err_
/// using embedded 4th-order method.
//------------------------------------------------------------------------------
template <class D>
void StepperDopr5<D>::dy(const double h, D& derivatives)
{
  static constexpr double c2 {0.2};
  static constexpr double c3 {0.3};
  static constexpr double c4 {0.8};
  static constexpr double c5 {8.0/9.0};
  static constexpr double a21 {0.2};
  static constexpr double a31 {3.0/40.0};
  static constexpr double a32 {9.0/40.0};
  static constexpr double a41 {44.0/45.0};
  static constexpr double a42 {-56.0/15.0};
  static constexpr double a43 {32.0/9.0};
  static constexpr double a51 {19372.0/6561.0};
  static constexpr double a52 {-25360.0/2187.0};
  static constexpr double a53 {64448.0/6561.0};
  static constexpr double a54 {-212.0/729.0};
  static constexpr double a61 {9017.0/3168.0};
  static constexpr double a62 {-355.0/33.0};
  static constexpr double a63 {46732.0/5247.0};
  static constexpr double a64 {49.0/176.0};
  static constexpr double a65 {-5103.0/18656.0};
  static constexpr double a71 {35.0/384.0};
  static constexpr double a72 {};
  static constexpr double a73 {500.0/1113.0};
  static constexpr double a74 {125.0/192.0};
  static constexpr double a75 {-2187.0/6784.0};
  static constexpr double a76 {11.0/84.0};

  static constexpr double e1 {71.0 / 57600.0};
  static constexpr double e3 {-71.0 / 16695.0};
  static constexpr double e4 {71.0 / 1920.0};
  static constexpr double e5 {-17253.0 / 339200.0};
  static constexpr double e6 {22.0 / 525.0};
  static constexpr double e7 {-1.0 / 40.0};

  std::vector<double> ytemp(n_);

  // First step
  for (std::size_t i {0}; i < n_; ++i)
  {
    ytemp[i] = y_[i] + h * a21 * dydx_[i];
  }
  // Second step.
  derivatives(x_ + c2 * h, ytemp, k2_);

  for (std::size_t i {0}; i <n_; ++i)
  {
    ytemp[i] = y_[i] + h * (a31 * dydx_[i] + a32 * k2_[i]);
  }

  // Third step.
  derivatives(x_ + c3 * h, ytemp, k3_);

  for (std::size_t i {0}; i < n_; ++i)
  {
    ytemp[i] = y_[i] + h * (a41 * dydx_[i] + a42 * k2_[i] + a43 * k3_[i]);
  }

  // Fourth step.
  derivatives(x_ + c4 * h, ytemp, k4_);

  for (std::size_t i {0}; i < n_; ++i)
  {
    ytemp[i] = y_[i] + h * (
      a51 * dydx_[i] + a52 * k2_[i] + a53 * k3_[i] + a54 * k4_[i]);
  }

  // Fifth step.
  derivatives(x_ + c5 * h, ytemp, k5_);

  for (std::size_t i {0}; i < n_; ++i)
  {
    ytemp[i] = y_[i] +
      h * (
        a61 * dydx_[i] +
        a62 * k2_[i] +
        a63 * k3_[i] +
        a64 * k4_[i] +
        a65 * k5_[i]);
  }

  const double xph {x_ + h};

  // Sixth step.
  derivatives(xph, ytemp, k6_);

  // Accumulate increments with proper weights.
  for (std::size_t i {0}; i < n_; ++i)
  {
    y_out_[i] = y_[i] + h * (
      a71 * dydx_[i] +
      a73 * k3_[i] +
      a74 * k4_[i] + 
      a75 * k5_[i] +
      a76 * k6_[i]);
  }

  // Will also be first evaluation for next step.
  derivatives(xph, y_out_, dydxnew_);

  for (std::size_t i {0}; i < n_; ++i)
  {
    // Estimate error as difference between fourth- and fifth-order methods.
    y_err_[i] = h * (
      e1 * dydx_[i] +
      e3 * k3_[i] +
      e4 * k4_[i] +
      e5 * k5_[i] +
      e6 * k6_[i] +
      e7 * dydxnew_[i]);
  }
}

//------------------------------------------------------------------------------
/// \details The routine prepare_dense uses the coefficients to set up the dense
/// output quantities.
/// \ref pp. 918 Ch. 17.2 Adaptive Stepsize Control for Runge-Kutta
/// \brief Store coefficients of interpolating polynomial for dense output in
/// rcont1...rcont5.
//------------------------------------------------------------------------------
template <class D>
void StepperDopr5<D>::prepare_dense(const double h, D& derivatives)
{
  std::vector<double> ytemp (n_);
  static constexpr double d1 {-12715105075.0 / 11282082432.0};

  for (int i {0}; i < n_; ++i)
  {
    rcont1_[i] = y_[i];
    const double ydiff = yout_[i] - y_[i];
    rcont2_[i] = ydiff;
    const double bsp1 = h * dydx_[i] - ydiff;
    rcont3_[i] = bsp1;
  }
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
  for (int i {0}; i < n_; ++i)
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
