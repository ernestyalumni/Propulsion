#include "ACoefficients.h"
#include "DOPRI5Coefficients.h"

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{
namespace DOPRI5Coefficients
{

const Coefficients::ACoefficients<7> a_coefficients {
  0.2,
  3.0 / 40.0,
  9.0 / 40.0,
  44.0 / 45.0,
  -56.0 / 15.0,
  32.0 / 9.0,
  19372.0 / 6561.0,
  -25360.0 / 2187.0,
  64448.0 / 6561.0,
  -212.0 / 729.0,
  9017.0 / 3168.0,
  -355.0 / 33.0,
  46732.0 / 5247.0,
  49.0 / 176.0,
  -5103.0 / 18656.0,
  35.0 / 384.0,
  0.0,
  500.0 / 1113.0,
  125.0 / 192.0,
  -2187.0 / 6784.0,
  11.0 / 84.0
};

} // namespace Coefficients  
} // namespace RKMethods
} // namespace ODE
} // namespace Numerical
