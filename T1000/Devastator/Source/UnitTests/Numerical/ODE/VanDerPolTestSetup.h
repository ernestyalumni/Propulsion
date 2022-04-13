#ifndef UNIT_TESTS_NUMERICAL_ODE_VAN_DER_POL_TEST_SETUP_H
#define UNIT_TESTS_NUMERICAL_ODE_VAN_DER_POL_TEST_SETUP_H

#include <vector>

namespace GoogleUnitTests
{
namespace Numerical
{
namespace ODE
{

namespace VanDerPol
{

inline const std::vector<double> y1_in_20_steps {
  2.,
  1.99001531,
  1.96373712,
  1.92544337,
  1.87808283,
  1.82343095,
  1.76269155,
  1.69650477, 
  1.62508332,
  1.54840951,
  1.4662393,  
  1.37809158,
  1.2833223,
  1.18102747, 
  1.07002585, 
  0.94885892,
  0.81579091, 
  0.66879173,
  0.50536802,
  0.3228295};

inline const std::vector<double> y2_in_20_steps {
  0.,
  -0.18038269,
  -0.31223354, 
  -0.41058163, 
  -0.48687504, 
  -0.54929801,
  -0.60359291,
  -0.65386198,
  -0.70312129,
  -0.7538284,
  -0.80801865,
  -0.86769084,
  -0.93477138,
  -1.01145789,
  -1.10029482,
  -1.2041735,
  -1.32633209,
  -1.47028169,
  -1.63899526,
  -1.83348537};

inline const std::vector<double> y1_result_20_steps {
  0.0,
  -0.1803826937734485,
  -0.31223354205225273,
  -0.41058163039002404,
  -0.48687504373875645,
  -0.5492980091298005,
  -0.6035929137597037,
  -0.6538619784127986,
  -0.7031212872392018,
  -0.75382839859263,
  -0.808018652496587,
  -0.8676908411735249,
  -0.9347713809593777,
  -1.0114578872672366,
  -1.1002948242881245,
  -1.204173504990997,
  -1.3263320911227412,
  -1.4702816858663437,
  -1.6389952598842674,
  -1.8334853719079132};

inline const std::vector<double> y2_result_20_steps {
  -2.0,
  -1.4560535077745809,
  -1.0719158570157303,
  -0.8138625156690724,
  -0.6476545930581228,
  -0.5463677693558335,
  -0.4908720877449637,
  -0.4684680017918621,
  -0.4713345536666331,
  -0.4948800410725622,
  -0.5371328441398118,
  -0.5979191525565629,
  -0.6786036164312667,
  -0.7816777111824667,
  -0.9105322644951261,
  -1.0688749809159528,
  -1.2594293531401783,
  -1.4814423256538596,
  -1.7257690781206694,
  -1.9652310683181158};

} // namespace VanDerPol

} // namespace ODE
} // namespace Numerical
} // namespace GoogleUnitTests

#endif // UNIT_TESTS_NUMERICAL_ODE_VAN_DER_POL_TEST_SETUP_H