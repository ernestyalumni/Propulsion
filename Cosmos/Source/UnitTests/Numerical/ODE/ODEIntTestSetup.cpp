#include "ODEIntTestSetup.h"

#include <vector>

using std::vector;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{

namespace ODEInt
{

void TestStepper::DerivativeType::operator()(
  const double,
  vector<double>&,
  vector<double>&)
{}

} // namespace ODEInt

} // namespace ODE
} // namespace Numerical
} // namespace GoogleUnitTests
