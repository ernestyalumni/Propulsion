#ifndef UNIT_TESTS_NUMERICAL_ODE_ODE_INT_TEST_SETUP_H
#define UNIT_TESTS_NUMERICAL_ODE_ODE_INT_TEST_SETUP_H

#include "Numerical/ODE/StepperBase.h"

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{

namespace ODEInt
{

class TestStepper : public ::Numerical::ODE::StepperBase
{
  public:

    using ::Numerical::ODE::StepperBase::StepperBase;

    class DerivativeType
    {
      public:

        void operator()(
          const double x,
          std::vector<double>& y,
          std::vector<double>& dydx);
    };
};

} // namespace ODEInt

} // namespace ODE
} // namespace Numerical
} // namespace GoogleUnitTests

#endif // UNIT_TESTS_NUMERICAL_ODE_ODE_INT_TEST_SETUP_H
