#include "Numerical/ODE/RKMethods/Coefficients/DOPRI5Coefficients.h"
#include "Numerical/ODE/RKMethods/Coefficients/KCoefficients.h"

#include "gtest/gtest.h"

#include <array>
#include <vector>

using Numerical::ODE::RKMethods::Coefficients::KCoefficients;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{
namespace RKMethods
{
namespace Coefficients
{

constexpr std::size_t S {::Numerical::ODE::RKMethods::DOPRI5Coefficients::s};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestKCoefficients, DefaultConstructs)
{
  {
    KCoefficients<S, std::vector<double>> ks;

    EXPECT_EQ(ks.size(), S);
    EXPECT_EQ(ks.get_ith_coefficient(1).size(), 0);
    EXPECT_EQ(ks.get_ith_coefficient(2).size(), 0);
    EXPECT_EQ(ks.get_ith_coefficient(S).size(), 0);
  }
  {
    KCoefficients<S, std::array<double, 5>> ks;

    EXPECT_EQ(ks.size(), S);
    EXPECT_EQ(ks.get_ith_coefficient(1).size(), 5);
    EXPECT_EQ(ks.get_ith_coefficient(2).size(), 5);
    EXPECT_EQ(ks.get_ith_coefficient(S).size(), 5);
  }  
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestKCoefficients, ConstructsWithPartialList)
{
  KCoefficients<S, std::vector<double>> ks {{0.1}, {2.2}, {33.3}};

  EXPECT_EQ(ks.size(), S);
  EXPECT_EQ(ks.get_ith_coefficient(1).at(0), 0.1);
  EXPECT_EQ(ks.get_ith_coefficient(2).at(0), 2.2);
  EXPECT_EQ(ks.get_ith_coefficient(S - 1).size(), 0);
  EXPECT_EQ(ks.get_ith_coefficient(S).size(), 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestKCoefficients, DefaultConstructsWithCompoundObjects)
{
  KCoefficients<S, std::vector<double>> ks {};

  EXPECT_EQ(ks.size(), S);
  EXPECT_EQ(ks[0].size(), 0);
  EXPECT_EQ(ks[1].size(), 0);
  EXPECT_EQ(ks[S - 1].size(), 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TestKCoefficients, PassingStdArrayAsReferenceToScalarMultiplyWorks)
{
  std::array<double, 1> out {};
  EXPECT_EQ(out[0], 0);

  KCoefficients<4, std::vector<double>> ks {
    {0.75},
    {0.90625},
    {0.9453125},
    {1.09765625}};

  ks.scalar_multiply(out, 1, 2.0);
  EXPECT_DOUBLE_EQ(out[0], 1.5);

  ks.scalar_multiply(out, 2, 2.0);
  EXPECT_DOUBLE_EQ(out[0], 1.8125);

  ks.scalar_multiply(out, 4, 2.0);
  EXPECT_DOUBLE_EQ(out[0], 2.1953125);
}

} // namespace Coefficients
} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests