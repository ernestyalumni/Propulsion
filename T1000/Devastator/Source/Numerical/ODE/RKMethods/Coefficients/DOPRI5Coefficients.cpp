#include "ACoefficients.h"
#include "BCoefficients.h"
#include "CCoefficients.h"
#include "DOPRI5Coefficients.h"

#include <cstddef>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{
namespace DOPRI5Coefficients
{

const Coefficients::ACoefficients<s> a_coefficients {
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

const Coefficients::CCoefficients<s> c_coefficients {
  0.2,
  0.3,
  0.8,
  8.0 / 9.0,
  1.0,
  1.0
};

const Coefficients::DeltaCoefficients<s> delta_coefficients {
  71.0 / 57600.0,
  0.0,
  -71.0 / 16695.0,
  71.0 / 1920.0,
  -17253.0 / 339200.0,
  22.0 / 525.0,
  -1.0 / 40.0
};

} // namespace Coefficients  
} // namespace RKMethods
} // namespace ODE
} // namespace Numerical
