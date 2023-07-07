#ifndef UTILITIES_FILEIO_TURBULENT_FLOW_CONFIGURATION_H
#define UTILITIES_FILEIO_TURBULENT_FLOW_CONFIGURATION_H

#include <cstddef> // std::size_t
#include <optional>
#include <string>
#include <unordered_map>

namespace Utilities
{
namespace FileIO
{

namespace TurbulentFlowConfiguration
{

struct StdSizeTParameters
{

  //------------------------------------------------------------------------
  /// Number of cells
  //------------------------------------------------------------------------
  std::optional<std::size_t> imax_;
  std::optional<std::size_t> jmax_;

  //------------------------------------------------------------------------
  /// Pressure
  /// itermax: maximum number of iterations for pressure per time step.
  //------------------------------------------------------------------------
  std::optional<std::size_t> itermax_;

  //------------------------------------------------------------------------
  /// Number of processes in x and y direction
  //------------------------------------------------------------------------
  std::optional<std::size_t> i_processes_;
  std::optional<std::size_t> j_processes_;

  StdSizeTParameters();

  ~StdSizeTParameters() = default;
};

std::unordered_map<std::string, std::optional<std::size_t>*>
  create_std_size_t_parameters_map(StdSizeTParameters& parameters);

struct IntTypeParameters
{
  // Solver type.
  std::optional<int> solver_;

  // Turbulence model.
  std::optional<int> model_;

  // Simulation type.
  std::optional<int> simulation_;

  std::optional<int> refine_;

  std::optional<int> preconditioner_;

  IntTypeParameters();

  ~IntTypeParameters() = default;
};

std::unordered_map<std::string, std::optional<int>*>
  create_int_type_parameters_map(IntTypeParameters& parameters);

struct DoubleTypeParameters
{
  //----------------------------------------------------------------------------
  /// Size of the domain
  //----------------------------------------------------------------------------
  std::optional<double> xlength_;      
  std::optional<double> ylength_;

  //----------------------------------------------------------------------------
  /// Time steps
  /// dt: time step size
  /// t_end: final time
  /// tau: safety factor for time step size control
  //----------------------------------------------------------------------------
  std::optional<double> dt_;
  std::optional<double> t_end_;
  std::optional<double> tau_;

  //----------------------------------------------------------------------------
  /// Output
  /// dt_value: time interval for writing files.
  //----------------------------------------------------------------------------
  std::optional<double> dt_value_;

  //----------------------------------------------------------------------------
  /// Pressure
  /// eps: tolerance for pressure iteration (residual < eps), i.e. accuracy
  /// bound for pressure.
  /// omg: relaxation factor for SOR
  /// gamma: upwind differencing factor
  //----------------------------------------------------------------------------
  std::optional<double> eps_;
  std::optional<double> omg_;
  std::optional<double> gamma_;

  //----------------------------------------------------------------------------
  /// Reynolds number (for example, for Lid Driven Cavity: 1 / nu)
  //----------------------------------------------------------------------------
  std::optional<double> re_;

  //----------------------------------------------------------------------------
  /// Gravity / External Forces  
  //----------------------------------------------------------------------------

  // gravitation x-direction
  std::optional<double> gx_;

  // gravitation x-direction
  std::optional<double> gy_;

  //----------------------------------------------------------------------------
  /// Initial Pressure
  //----------------------------------------------------------------------------
  std::optional<double> pi_;

  //----------------------------------------------------------------------------
  /// Initial Velocity
  //----------------------------------------------------------------------------
  // Velocity x-direction
  std::optional<double> ui_;

  // Velocity y-direction
  std::optional<double> vi_;

  // Viscosity
  std::optional<double> nu_;

  // Prandtl number
  std::optional<double> pr_;

  // Thermal expansion coefficient
  std::optional<double> beta_;

  // Thermal diffusivity
  std::optional<double> alpha_;

  DoubleTypeParameters();

  ~DoubleTypeParameters() = default;

  //----------------------------------------------------------------------------
  /// \details Do the following:
  /// If viscosity nu isn't given yet, but Reynolds number Re is, set
  /// nu = 1 / Re
  /// otherwise, if both are not set yet, then default the viscosity to 0.0.
  /// If alpha, thermal diffusivity, isn't set yet, but Prandtl number Pr is and
  /// viscosity nu is, then using Prandtl number = nu / alpha,
  /// alpha = nu / pr
  //----------------------------------------------------------------------------
  void initialize();
};

std::unordered_map<std::string, std::optional<double>*>
  create_double_type_parameters_map(DoubleTypeParameters& parameters);

struct UnorderedMapTypeParameters
{
  std::unordered_map<std::size_t, double> wall_temperatures_;

  std::unordered_map<std::size_t, double> wall_velocities_;

  // Velocities U, V in x- and y- directions, respectively for the inlet.

  std::unordered_map<std::size_t, double> inlet_Us_;
  std::unordered_map<std::size_t, double> inlet_Vs_;

  std::unordered_map<std::size_t, double> inlet_Ts_;

  // Inlet turbulent kinetic energy (k)
  std::unordered_map<std::size_t, double> inlet_Ks_;

  // Inlet rate of dissipation of turbulence kinetic energy (\epsilon) due to
  // viscosity.
  std::unordered_map<std::size_t, double> inlet_eps_;

  UnorderedMapTypeParameters() = default;

  ~UnorderedMapTypeParameters() = default;
};

} // namespace TurbulentFlowConfiguration
} // namespace FileIO
} // namespace Utilities

#endif // UTILITIES_FILEIO_TURBULENT_FLOW_CONFIGURATION_H