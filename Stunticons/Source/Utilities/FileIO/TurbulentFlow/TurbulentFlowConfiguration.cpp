#include "TurbulentFlowConfiguration.h"

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>

using std::nullopt;
using std::optional;
using std::string;
using std::unordered_map;

namespace Utilities
{
namespace FileIO
{
namespace TurbulentFlowConfiguration
{

StdSizeTParameters::StdSizeTParameters():
  imax_{nullopt},
  jmax_{nullopt},
  itermax_{nullopt},
  i_processes_{nullopt},
  j_processes_{nullopt}
{}

unordered_map<string, optional<std::size_t>*> create_std_size_t_parameters_map(
  StdSizeTParameters& parameters)
{
  return unordered_map<string, optional<std::size_t>*> {
    {"imax", &parameters.imax_},
    {"jmax", &parameters.jmax_},
    {"itermax", &parameters.itermax_},
    {"iproc", &parameters.i_processes_},
    {"jproc", &parameters.j_processes_}
  };
}

IntTypeParameters::IntTypeParameters():
  solver_{nullopt},
  model_{nullopt},
  simulation_{nullopt},
  refine_{nullopt},
  preconditioner_{nullopt}
{}

unordered_map<string, optional<int>*> create_int_type_parameters_map(
  IntTypeParameters& parameters)
{
  return unordered_map<string, optional<int>*> {
    {"solver", &parameters.solver_},
    {"model", &parameters.model_},
    {"simulation", &parameters.simulation_},
    {"refine", &parameters.refine_},
    {"preconditioner", &parameters.preconditioner_}
  };
}

DoubleTypeParameters::DoubleTypeParameters():
  xlength_{nullopt},
  ylength_{nullopt},
  dt_{nullopt},
  t_end_{nullopt},
  tau_{nullopt},
  dt_value_{nullopt},
  eps_{nullopt},
  omg_{nullopt},
  gamma_{nullopt}
{}

void DoubleTypeParameters::initialize()
{
  if (!nu_.has_value() && re_.has_value())
  {
    // min is the smallest positive value for a double type.
    if (std::abs(*re_) > std::numeric_limits<double>::min() * 2)
    {    
      *nu_ = 1.0 / *re_;
    }
  }
  else if (!re_.has_value() && !nu_.has_value())
  {
    *nu_ = 0.0;
  }

  if (!alpha_.has_value() && pr_.has_value() && nu_.has_value())
  {
    if (std::abs(*pr_) > std::numeric_limits<double>::min() * 2)
    {
      *alpha_ = *nu_ / *pr_;
    }
  }
  else if (!alpha_.has_value())
  {
    *alpha_ = 0.0;
    if (!beta_.has_value())
    {
      *beta_ = 0.0;
    }
  }
}

unordered_map<string, optional<double>*> create_double_type_parameters_map(
  DoubleTypeParameters& parameters)
{
  return unordered_map<string, optional<double>*> {
    {"xlength", &parameters.xlength_},
    {"ylength", &parameters.ylength_},
    {"dt", &parameters.dt_},
    {"t_end", &parameters.t_end_},
    {"tau", &parameters.tau_},
    {"dt_value", &parameters.dt_value_},
    {"eps", &parameters.eps_},
    {"omg", &parameters.omg_},
    {"gamma", &parameters.gamma_},
    {"Re", &parameters.re_},
    {"GX", &parameters.gx_},
    {"GY", &parameters.gy_},
    {"PI", &parameters.pi_},
    {"UI", &parameters.ui_},
    {"VI", &parameters.vi_},
    {"nu", &parameters.nu_},
    {"Pr", &parameters.pr_},
    {"beta", &parameters.beta_},
    {"alpha", &parameters.alpha_}
  };
}

} // namespace TurbulentFlowConfiguration
} // namespace FileIO
} // namespace Utilities
