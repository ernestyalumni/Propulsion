#include "ACoefficients.h"
#include "BCoefficients.h"
#include "CCoefficients.h"
#include "RK4Coefficients.h"

#include <cstddef>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{
namespace RK4Coefficients
{

const Coefficients::ACoefficients<s> a_coefficients {
  0.5,
  0.0,
  0.5,
  0.0,
  0.0,
  1.0
};

const Coefficients::DeltaCoefficients<s> b_coefficients {
  1.0 / 6.0,
  2.0 / 6.0,
  2.0 / 6.0,
  1.0 / 6.0
};

const Coefficients::CCoefficients<s> c_coefficients {
  0.5,
  0.5,
  1.0
};

} // namespace RK4Coefficients  
} // namespace RKMethods
} // namespace ODE
} // namespace Numerical
