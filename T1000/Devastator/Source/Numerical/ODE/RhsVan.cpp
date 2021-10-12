#include "RhsVan.h"

#include <vector>

using std::vector;

namespace Numerical
{
namespace ODE
{

void RhsVan::operator()(
  const double,
  vector<double>& y,
  vector<double>& dydx)
{
  dydx[0] = y[1];
  dydx[1] = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / eps_;
}

} // namespace ODE
} // namespace Numerical
