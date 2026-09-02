//------------------------------------------------------------------------------
/// \file emit_pi_step_size_golden.cpp
/// \brief Emit golden vectors for the PI step-size controller from the C++
///   reference implementation, for the Rust twin to check against.
///
/// Build and run from the repository root:
///   g++ -std=c++17 -O2 -I Cosmos/Source \
///       Cosmos/Rust/tools/emit_pi_step_size_golden.cpp -o /tmp/emit_pi_golden
///   /tmp/emit_pi_golden      > Cosmos/Rust/golden/pi_step_size.json
///   /tmp/emit_pi_golden tsv  > Cosmos/Rust/golden/pi_step_size.tsv
//------------------------------------------------------------------------------
#include "Numerical/ODE/RKMethods/ComputePIStepSize.h"

#include <cstdio>
#include <string>
#include <vector>

using Numerical::ODE::RKMethods::ComputePIStepSize;

struct ControllerParameters
{
  double alpha;
  double beta;
  double min_scale;
  double max_scale;
  double safety_factor;
};

int main(int argc, char** argv)
{
  const bool as_tsv {argc > 1 && std::string{argv[1]} == "tsv"};

  // The first set is what Cosmos' DOPRI5 tests use (TestSetup.h: alpha_5 =
  // 0.7 / 5, beta_5 = 0.08) with the header defaults for the other three.
  // The second and third exercise non-default bounds and safety factor.
  const std::vector<ControllerParameters> parameter_sets {
    {0.7 / 5.0, 0.08, 0.2, 5.0, 0.9},
    {0.2, 0.0, 0.2, 10.0, 0.9},
    {0.7 / 8.0, 0.4 / 8.0, 0.333, 6.0, 0.8}};

  const std::vector<double> errors {
    0.0, 1.0e-12, 1.0e-6, 1.0e-3, 0.1, 0.5, 0.999, 1.0, 1.0 + 1.0e-12, 1.5, 4.0,
    100.0, 1.0e6};
  const std::vector<double> previous_errors {1.0e-12, 1.0e-4, 0.3, 1.0, 7.0};
  const std::vector<double> steps {1.0e-9, 1.0e-3, 1.0, 60.0};

  if (as_tsv)
  {
    std::printf(
      "# source: Cosmos/Source/Numerical/ODE/RKMethods/ComputePIStepSize.h\n");
    std::printf(
      "alpha\tbeta\tmin_scale\tmax_scale\tsafety_factor\terror\t"
      "previous_error\th\twas_rejected\th_new\n");
  }
  else
  {
    std::printf("{\n  \"source\": "
      "\"Cosmos/Source/Numerical/ODE/RKMethods/ComputePIStepSize.h\",\n");
    std::printf("  \"cases\": [\n");
  }

  bool first {true};
  for (const auto& p : parameter_sets)
  {
    ComputePIStepSize<double> controller {
      p.alpha, p.beta, p.min_scale, p.max_scale, p.safety_factor};

    for (const double error : errors)
    {
      for (const double previous_error : previous_errors)
      {
        for (const double h : steps)
        {
          for (const bool was_rejected : {false, true})
          {
            const double h_new {
              controller.compute_new_step_size(
                error, previous_error, h, was_rejected)};

            if (as_tsv)
            {
              std::printf(
                "%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%d\t"
                "%.17g\n",
                p.alpha, p.beta, p.min_scale, p.max_scale, p.safety_factor,
                error, previous_error, h, was_rejected ? 1 : 0, h_new);
              continue;
            }

            std::printf(
              "%s    {\"alpha\": %.17g, \"beta\": %.17g, \"min_scale\": %.17g, "
              "\"max_scale\": %.17g, \"safety_factor\": %.17g, "
              "\"error\": %.17g, \"previous_error\": %.17g, \"h\": %.17g, "
              "\"was_rejected\": %s, \"h_new\": %.17g}",
              first ? "" : ",\n",
              p.alpha, p.beta, p.min_scale, p.max_scale, p.safety_factor,
              error, previous_error, h, was_rejected ? "true" : "false", h_new);
            first = false;
          }
        }
      }
    }
  }

  if (!as_tsv)
  {
    std::printf("\n  ]\n}\n");
  }
  return 0;
}
