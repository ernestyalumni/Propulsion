#ifndef NUMERICAL_ODE_ODE_INTEGRATION_H
#define NUMERICAL_ODE_ODE_INTEGRATION_H

#include <vector>

namespace Numerical
{
namespace ODE
{

template <class Stepper, class Output>
class OdeIntegration
{
  public:

    // Take at most MAXSTP steps.
    static constexpr std::size_t MAXSTP {50000};

    using Derivatives = Stepper::DerivativeType;

    OdeIntegration(
      std::vector<double>& y_start_t,
      const double xx1,
      const double xx2,
      const double a_tolerance,
      const double r_tolerance,
      const double h1,
      const double h_min_n,
      Output& outt,
      Stepper::DerivativeType& derivatives);

    void integrate();

  private:

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
    std::vector<double> dy_;
    std::vector<double> dx_;
    std::vector<double> ystart_;
    Output output_;

    Derivatives& derivatives_;

    Stepper s_;
    int n_stp_;
    double x_;
    double h_;
};

template <class Stepper>
OdeIntegration<Stepper>::OdeIntegration(
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
  n_var_(ystartt.size()),
  y_{n_var_},
  dydx_{n_var_},
  ystart_{ystartt},
  x_{xx1},
  nok_{0},
  nbad_{0},
  x1_{xx1},
  x2_{xx2},
  hmin_{hmin_n},
  dense_{outt.dense_},
  out_{outt},
  derivatives_{derivatives},
  s_{y_, dydx_, x_, a_tolerance, r_tolerance, dense}
{
  EPS_ = std::numeric_limits<double>::epsilon();
  h = SIGN(h1_, x2_ - x1_);
  for (int i {0}; i < n_var_; ++i)
  {
    y_[i] = ystart_[i];
  }
  out_.init(s_.n_eqns_, x1_, x2_);
}

template <class Stepper>
void OdeIntegration<Stepper>::integrate()
{
  derivatives_(x, y, dydx);
  if (dense)
  {
    out_.out(-1, x, y, s, h);
  }
  else
  {
    out_.save(x, y);
  }

  for (n_stp_ = 0; n_stp_ < MAXSTP; ++n_stp_)
  {
    // If stepsize can oversheet, decrease.
    if ((x_ + h_ * 1.0001 -x2_) * (x2_ - x1_) > 0.0)
    {
      h_ = x2_ - x_
    }
    // Take a step.
    s_.step(h, derivatives_);

    if (s_.hdid_ == h_)
    {
      ++nok_;
    }
    else
    {
      ++nbad_;
    }

    if (dense)
    {
      out_.out(n_stp_, x_, y_, s_, s_.hdid_);
    }
    else
    {
      out_.save(x_, y_);
    }

    if ((x_ - x2_) * (x2_ - x1_) >= 0.0)
    {
      for (int i {0}; i < n_var_; ++i)
      {
        ystart_[i] = y_[i];
      }
      if (
        out_.kmax_ > 0 &&
          std::abs(out_.xsave[out_.count_ - 1] - x2_) >
            100.0 * std::abs(x2_) * EPS_)
      {
        out_.save(x_, y_);
      }
      return;
    }
    if (std::abs(s_.hnext_) <= hmin_)
    {
      throw("Step size too small in OdeIntegration");
    }
    h_ * s_.hnext_;
  }

  throw("Too many steps in routine OdeIntegration");
}

} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_ODE_INTEGRATION_H
