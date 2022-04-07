#include "Numerical/ODE/RKMethods/CalculateNewYAndError.h"
#include "Numerical/ODE/RKMethods/Coefficients/DOPRI5Coefficients.h"

#include "gtest/gtest.h"

#include <algorithm> // std::transform
#include <iterator>
#include <vector>

using Numerical::ODE::RKMethods::CalculateNewYAndError;
using std::back_inserter;
using std::transform;
using std::vector;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

const auto& DOPRI5_s = ::Numerical::ODE::RKMethods::DOPRI5Coefficients::s;

const auto& DOPRI5_a_coefficients =
  ::Numerical::ODE::RKMethods::DOPRI5Coefficients::a_coefficients;

const auto& DOPRI5_c_coefficients =
  ::Numerical::ODE::RKMethods::DOPRI5Coefficients::c_coefficients;

const auto& DOPRI5_delta_coefficients =
  ::Numerical::ODE::RKMethods::DOPRI5Coefficients::delta_coefficients;

class Examplef
{
  public:

    Examplef() = default;

    void operator()(
      const double x,
      const vector<double>& y,
      vector<double>& output)
    {
      transform(
        y.begin(),
        y.end(),
        back_inserter(output),
        [x](const double y_value)
        {
          return y_value - x * x + 1.0;
        }
      ); 
    }

    vector<double> operator()(
      const double x,
      const vector<double>& y)
    {
      vector<double> output;
      this->operator()(x, y, output);
      return output;
    }
};

auto examplef = [](
  const double x,
  const vector<double>& y,
  vector<double>& output)
{
  transform(
    y.begin(),
    y.end(),
    back_inserter(output),
    [x](const double y_value)
    {
      return y_value - x * x + 1.0;
    });
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewYAndError, ConstructsFromRValueDerivative)
{
  {
    CalculateNewYAndError new_y_and_err {
      Examplef{},
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};
  }
  {
    CalculateNewYAndError<DOPRI5_s, Examplef> new_y_and_err {
      Examplef{},
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestCalculateNewYAndError, ConstructsFromLValueDerivative)
{
  Examplef f {};
  {
    CalculateNewYAndError new_y_and_err {
      f,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};
  }
  {
    CalculateNewYAndError<DOPRI5_s, Examplef> new_y_and_err {
      f,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};
  }
  {
    CalculateNewYAndError<DOPRI5_s, decltype(f)> new_y_and_err {
      f,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};
  }
  {
    CalculateNewYAndError new_y_and_err {
      examplef,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};
  }
  {
    CalculateNewYAndError<DOPRI5_s, decltype(examplef)> new_y_and_err {
      examplef,
      DOPRI5_a_coefficients,
      DOPRI5_c_coefficients,
      DOPRI5_delta_coefficients};
  }
}

} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests