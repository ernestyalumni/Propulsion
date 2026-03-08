#ifndef NUMERICAL_ODE_RK4_H
#define NUMERICAL_ODE_RK4_H

#include <functional>
#include <vector>

namespace Numerical
{
namespace ODE
{

void rk4(
  std::vector<double>& y,
  std::vector<double>& dydx,
  const double x,
  const double h,
  std::vector<double>& yout,
  std::function<void (
    const double,
    std::vector<double>&,
    std::vector<double>&)> derivative
  );

} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK4_H
