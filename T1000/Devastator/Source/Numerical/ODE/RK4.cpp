#include "RK4.h"

#include <cstddef>
#include <vector>

using std::size_t;
using std::vector;

namespace Numerical
{
namespace ODE
{

void rk4(
  vector<double>& y,
  vector<double>& dydx,
  //const double x,
  const double h,
  vector<double>& yout)
{
  const size_t n {y.size()};
  vector<double> dym(n);
  vector<double> dyt(n);
  vector<double> yt(n);

  const double hh {h * 0.5};
  const double h6 {h / 6.0};
  //const double xh {x + hh};

  for (size_t i {0}; i < n; ++i)
  {
    yt[i] = y[i] + hh * dydx[i];
  }

  for (size_t i {0}; i < n; ++i)
  {
    yt[i] = y[i] + hh * dyt[i];
  }

  for (size_t i {0}; i < n; ++i)
  {
    yt[i] = y[i] + h * dym[i];
    dym[i] += dyt[i];
  }

  // Fourth step.

  for (size_t i {0}; i < n; ++i)
  {
    // Accumulate increments with proper weights.
    yout[i] = y[i] + h6 * (dydx[i] + dyt[i] + 2.0 * dym[i]);
  }

}

} // namespace ODE
} // namespace Numerical
