#include "TestSetup.h"
#include "gtest/gtest.h"

#include <valarray>
#include <vector>

using std::valarray;
using std::vector;

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ForcedOscillationsSetupTest, ExactSolutionGivesExactSolution)
{
  ForcedOscillationExactSolution ex {};
  EXPECT_DOUBLE_EQ(ex.compute_exact_solution(0.0), 0.0);
  EXPECT_DOUBLE_EQ(ex.compute_exact_solution(20.0), 0.952533660328207);
  EXPECT_NEAR(ex.compute_exact_solution(40.0), -0.424905155163017, 1e-15);
  EXPECT_DOUBLE_EQ(ex.compute_exact_solution(60.0), -1.07509638224269);
  EXPECT_DOUBLE_EQ(ex.compute_exact_solution(80.0), 0.669750999207618);
  EXPECT_DOUBLE_EQ(ex.compute_exact_solution(100.0), 0.923281796525534);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ForcedOscillationsSetupTest, EquationOfMotionWorksWithStdVector)
{
  {
    const auto result = forced_oscillation_eq_of_motion<vector<double>>(
      0.0,
      vector<double>{1.0, 2.0});

    EXPECT_DOUBLE_EQ(result[0], 2.0);
    EXPECT_DOUBLE_EQ(result[1], -1.12);
  }
  
  const vector<double> inputs {0.0, 0.0};
  {
    const auto result = forced_oscillation_eq_of_motion<vector<double>>(
      20.0,
      inputs);

    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 0.989358246623382);
  }
  {
    const auto result = forced_oscillation_eq_of_motion<vector<double>>(
      40.0,
      inputs);

    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_NEAR(result[1], -0.287903316665065, 1e-15);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ForcedOscillationsSetupTest, EquationOfMotionWorksWithStdValarray)
{
  {
    const auto result = forced_oscillation_eq_of_motion<valarray<double>>(
      0.0,
      valarray<double>{1.0, 2.0});

    EXPECT_DOUBLE_EQ(result[0], 2.0);
    EXPECT_DOUBLE_EQ(result[1], -1.12);
  }
  
  const valarray<double> inputs {0.0, 0.0};
  {
    const auto result = forced_oscillation_eq_of_motion<valarray<double>>(
      20.0,
      inputs);

    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 0.989358246623382);
  }
  {
    const auto result = forced_oscillation_eq_of_motion<valarray<double>>(
      40.0,
      inputs);

    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_NEAR(result[1], -0.287903316665065, 1e-15);
  }
}


} // namespace RKMethods
} // namespace ODE 
} // namespace Numerical
} // namespace GoogleUnitTests