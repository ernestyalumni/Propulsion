#include "Numerical/ODE/RKMethods/IntegrateWithPIControl.h"
#include "Numerical/ODE/RKMethods/Coefficients/DOPRI5Coefficients.h"
#include "gtest/gtest.h"
#include <array>
#include <cmath>
#include <limits>

namespace {
namespace RK = Numerical::ODE::RKMethods;
namespace DP = RK::DOPRI5Coefficients;
using State = std::array<double, 1>;
struct Example {
  State operator()(double t, const State& y) const { return {y[0]-t*t+1}; }
};
double exact(double t) { return (t+1)*(t+1)-0.5*std::exp(t); }
template<class RHS>
auto calculation(RHS rhs) {
  return RK::CalculateNewYAndError<DP::s, RHS>{std::move(rhs),
    DP::a_coefficients, DP::c_coefficients, DP::delta_coefficients};
}
template<class RHS>
auto driver(RHS rhs, std::size_t max_steps=50000) {
  return RK::IntegrateWithPIControl{calculation(std::move(rhs)),
    RK::CalculateScaledError{1e-10,1e-10}, RK::ComputePIStepSize{0.14,0.08}, max_steps};
}

TEST(CosmosODEContract, ArrayStepPreservesReferenceErrorAndFSAL) {
  auto calc=calculation(Example{});
  RK::Coefficients::KCoefficients<7,State> stages;
  const State initial{0.5};
  const State y=calc.calculate_new_y(0.5,0.0,initial,State{1.5},stages);
  const State error=calc.calculate_error(0.5,stages);
  EXPECT_NEAR(y[0],exact(0.5),1e-5);
  EXPECT_NEAR(error[0],-2.4370659722241367e-5,1e-13);
  EXPECT_DOUBLE_EQ(stages.get_ith_coefficient(7)[0],Example{}(0.5,y)[0]);
}

TEST(CosmosODEContract, RepeatedIntegrationResetsControllerAndEndsExactly) {
  auto solver=driver(Example{});
  const RK::IntegrationInputs inputs{State{0.5},0.0,3.0,0.1};
  auto first=solver.integrate<1>(inputs);
  auto second=solver.integrate<1>(inputs);
  EXPECT_EQ(first,second);
  EXPECT_DOUBLE_EQ(std::get<0>(first).back(),3.0);
  EXPECT_NEAR(std::get<1>(first).back()[0],exact(3.0),1e-8);
}

TEST(CosmosODEContract, NoRhsCallOutsideIntervalAndLastAllowedTrialSucceeds) {
  using Six=std::array<double,6>;
  auto rhs=[](double t,const Six&) {
    if(t<0 || t>1) throw std::domain_error("Outside RHS domain");
    return Six{1,2,3,4,5,6};
  };
  auto solver=driver(rhs,1);
  int callbacks=0;
  auto result=solver.integrate_with_observer<6>(RK::IntegrationInputs{Six{},0.,1.,2.},
    [&](double t,const Six& y,double h) {
      ++callbacks; EXPECT_DOUBLE_EQ(t,1.); EXPECT_DOUBLE_EQ(h,1.);
      EXPECT_NEAR(y[5],6.,1e-13);
    },1);
  EXPECT_EQ(callbacks,1);
  EXPECT_EQ(std::get<2>(result),1u);
}

TEST(CosmosODEContract, BackwardIntegrationWorksWithDefaultAndPositiveHints) {
  auto solver=driver(Example{});
  for(double hint:{0.,0.2}) {
    auto result=solver.integrate<1>(RK::IntegrationInputs{State{exact(2.)},2.,0.,hint});
    const auto& times=std::get<0>(result);
    EXPECT_DOUBLE_EQ(times.back(),0.);
    for(std::size_t i=1;i<times.size();++i) EXPECT_LT(times[i],times[i-1]);
    EXPECT_NEAR(std::get<1>(result).back()[0],0.5,1e-9);
  }
}

TEST(CosmosODEContract, EmptyIntervalDoesNotEvaluateOrObserve) {
  auto rhs=[](double,const State&)->State { throw std::runtime_error("RHS should not run"); };
  auto solver=driver(rhs,0);
  auto result=solver.integrate_with_observer<1>(RK::IntegrationInputs{State{4.},2.,2.},
    [](double,const State&,double) { FAIL()<<"Observer should not run"; });
  EXPECT_DOUBLE_EQ(std::get<0>(result),2.);
  EXPECT_EQ(std::get<1>(result),State{4.});
  EXPECT_EQ(std::get<2>(result),0u);
}

TEST(CosmosODEContract, ExhaustedStepBudgetIsAnError) {
  auto solver=driver(Example{},1);
  EXPECT_THROW(solver.integrate<1>(RK::IntegrationInputs{State{0.5},0.,3.,0.1}),std::runtime_error);
}

TEST(CosmosODEContract, FailedNonfiniteTrialDoesNotAdvanceAcceptedState) {
  auto rhs=[](double t,const State&) {
    return State{t>0 ? std::numeric_limits<double>::quiet_NaN() : 1.};
  };
  RK::StepWithPIControl step{calculation(rhs),RK::CalculateScaledError{1e-8,1e-8},
    RK::ComputePIStepSize{0.14,0.08}};
  RK::StepInputs<7,State> inputs{State{2.},State{1.},0.1,0.};
  EXPECT_THROW(step.step<1>(inputs),std::domain_error);
  EXPECT_DOUBLE_EQ(inputs.x_n_,0.);
  EXPECT_EQ(inputs.y_n_,State{2.});
  EXPECT_EQ(inputs.dydx_n_,State{1.});
}

TEST(CosmosODEContract, InvalidTimeAndUnrepresentableStepAreErrors) {
  auto solver=driver(Example{});
  EXPECT_THROW(solver.integrate<1>(RK::IntegrationInputs{State{0.5},0.,
    std::numeric_limits<double>::infinity(),0.1}),std::invalid_argument);
  RK::StepWithPIControl step{calculation(Example{}),RK::CalculateScaledError{1e-8,1e-8},
    RK::ComputePIStepSize{0.14,0.08}};
  RK::StepInputs<7,State> inputs{State{1.},State{1.},1.,1e20};
  EXPECT_THROW(step.step<1>(inputs),std::runtime_error);
  inputs.h_n_=0.;
  EXPECT_THROW(step.step<1>(inputs),std::invalid_argument);
}

TEST(CosmosODEContract, ArrayKernelSupportsFloatWithDifferentStageAndStateCounts) {
  using Three=std::array<float,3>;
  auto rhs=[](float,const Three& y) { return y; };
  const RK::Coefficients::ACoefficients<2,float> a{1.f};
  const RK::Coefficients::CCoefficients<2,float> c{1.f};
  const RK::Coefficients::DeltaCoefficients<2,float> delta{0.f,0.f};
  RK::CalculateNewYAndError<2,decltype(rhs),float> calc{rhs,a,c,delta};
  RK::Coefficients::KCoefficients<2,Three> stages;
  auto y=calc.calculate_new_y(0.5f,0.f,Three{1,2,3},Three{1,2,3},stages);
  EXPECT_EQ(y,(Three{1.5f,3.f,4.5f}));
}
} // namespace
