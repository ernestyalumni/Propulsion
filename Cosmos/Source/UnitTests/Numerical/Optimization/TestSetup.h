#ifndef GOOGLE_UNIT_TESTS_NUMERICAL_OPTIMIZATION_TEST_SETUP_H
#define GOOGLE_UNIT_TESTS_NUMERICAL_OPTIMIZATION_TEST_SETUP_H

namespace GoogleUnitTests
{
namespace Numerical
{
namespace Optimization
{

inline double example_f1(const double x)
{
  return x * x + 1.0;
};

inline double example_f2(const double x)
{
  return x * x - 4.0 * x + 4.0;
}

inline double example_f3(const double x)
{
  return -x;
};

} // namespace Optimization
} // namespace Numerical
} // namespace GoogleUnitTests

#endif // GOOGLE_UNIT_TESTS_NUMERICAL_OPTIMIZATION_TEST_SETUP_H