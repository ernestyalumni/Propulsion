//------------------------------------------------------------------------------
/// \file   wasm_bridge.cpp
/// \brief  Emscripten bridge: expose NumerovOrbit C++ to JavaScript/WASM.
///
/// \details This is a thin C API wrapper around the C++ template classes.
///   Emscripten compiles this to .wasm + .js glue.  The JavaScript side loads
///   the module and calls these extern "C" functions.
///
/// Build (from Cosmos/Source/Examples/Astrodynamics):
///   emcc wasm_bridge.cpp \
///     -I ../../../.. \
///     -o numerov_orbit.js \
///     -s WASM=1 \
///     -s EXPORTED_FUNCTIONS='["_malloc","_free"]' \
///     -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
///     -s ALLOW_MEMORY_GROWTH=1 \
///     -s MODULARIZE=1 \
///     -s EXPORT_NAME="NumerovOrbitModule" \
///     -s ENVIRONMENT=web \
///     -O3 \
///     -std=c++20
///
/// Or use the provided CMake target (see Examples/CMakeLists.txt).
//------------------------------------------------------------------------------

#include <emscripten.h>

#include "Astrodynamics/Propagators/NumerovOrbit.h"
#include "Astrodynamics/TwoBodyAcceleration.h"
#include "Astrodynamics/specific_energy.h"

#include <memory>

using Astrodynamics::Propagators::NumerovOrbit;
using Astrodynamics::Propagators::OrbitalState;
using Astrodynamics::TwoBodyAcceleration::TotalAcceleration;
using Astrodynamics::TwoBodyAcceleration::NewtonsGravitation;
using Astrodynamics::TwoBodyAcceleration::J2Perturbation;
using Astrodynamics::SpecificEnergy;
using Algebra::Modules::Vectors::Vector3;

//------------------------------------------------------------------------------
// C API opaque handle
//------------------------------------------------------------------------------
extern "C" {

//------------------------------------------------------------------------------
/// \brief Configuration struct passed from JS to create propagator.
///
/// Layout (32 bytes):
///   0-7   : mu      (m³/s²)
///   8-15  : j2      (dimensionless)
///   16-23 : r_earth (m)
///   24-31 : dt      (s)
//------------------------------------------------------------------------------
struct PropagatorConfig
{
  double mu;
  double j2;
  double r_earth;
  double dt;
};

//------------------------------------------------------------------------------
/// \brief Initial state passed from JS.
///
/// Layout (56 bytes):
///   0-7   : t  (s)
///   8-31  : r  (x, y, z) in meters
///   32-55 : v  (vx, vy, vz) in m/s
//------------------------------------------------------------------------------
struct InitialState
{
  double t;
  double rx, ry, rz;
  double vx, vy, vz;
};

//------------------------------------------------------------------------------
/// \brief Output state returned to JS.
///
/// Same layout as InitialState plus specific orbital energy.
//------------------------------------------------------------------------------
struct OutputState
{
  double t;
  double rx, ry, rz;
  double vx, vy, vz;
  double specific_energy;  // J/kg
};

//------------------------------------------------------------------------------
// Opaque handle type
//------------------------------------------------------------------------------
struct NumerovOrbitHandle
{
  std::unique_ptr<NumerovOrbit<double>> propagator;
  double mu;  // cached for energy calculation
};

//------------------------------------------------------------------------------
/// \brief Create a new orbit propagator.
///
/// \param config   Pointer to PropagatorConfig (must be 32 bytes).
/// \param initial  Pointer to InitialState (must be 56 bytes).
/// \param use_j2   Non-zero to include J2 perturbation, 0 for two-body only.
///
/// \return Opaque handle pointer, or nullptr on error.
///   Caller must pass handle to numerov_orbit_destroy() when done.
//------------------------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
NumerovOrbitHandle* numerov_orbit_create(
  const PropagatorConfig* config,
  const InitialState* initial,
  int use_j2)
{
  if (!config || !initial) return nullptr;

  try
  {
    // Build acceleration model
    TotalAcceleration<double> accel{};
    accel.add(NewtonsGravitation<double>{config->mu});
    if (use_j2)
    {
      accel.add(J2Perturbation<double>{config->mu, config->j2, config->r_earth});
    }

    // Initial orbital state
    Vector3<double> r0{initial->rx, initial->ry, initial->rz};
    Vector3<double> v0{initial->vx, initial->vy, initial->vz};
    OrbitalState<double> s0{initial->t, std::move(r0), std::move(v0)};

    // Create propagator (startup happens in constructor)
    auto handle = std::make_unique<NumerovOrbitHandle>();
    handle->propagator = std::make_unique<NumerovOrbit<double>>(
      std::move(accel), s0, config->dt);
    handle->mu = config->mu;

    return handle.release();
  }
  catch (...)
  {
    return nullptr;
  }
}

//------------------------------------------------------------------------------
/// \brief Advance the propagator by one fixed step.
///
/// \param handle  Opaque handle from numerov_orbit_create().
//------------------------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
void numerov_orbit_step(NumerovOrbitHandle* handle)
{
  if (!handle || !handle->propagator) return;
  handle->propagator->step();
}

//------------------------------------------------------------------------------
/// \brief Advance the propagator by N steps (batch for efficiency).
///
/// \param handle  Opaque handle from numerov_orbit_create().
/// \param n       Number of steps to advance.
//------------------------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
void numerov_orbit_step_n(NumerovOrbitHandle* handle, int n)
{
  if (!handle || !handle->propagator) return;
  for (int i = 0; i < n; ++i)
  {
    handle->propagator->step();
  }
}

//------------------------------------------------------------------------------
/// \brief Get current orbital state.
///
/// \param handle  Opaque handle from numerov_orbit_create().
/// \param out     Pointer to OutputState to fill (must be 64 bytes).
//------------------------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
void numerov_orbit_get_state(NumerovOrbitHandle* handle, OutputState* out)
{
  if (!handle || !handle->propagator || !out) return;

  const auto s = handle->propagator->get_state();
  out->t = s.t;
  out->rx = s.r.x();
  out->ry = s.r.y();
  out->rz = s.r.z();
  out->vx = s.v.x();
  out->vy = s.v.y();
  out->vz = s.v.z();

  // Compute specific orbital energy
  SpecificEnergy<double> energy{handle->mu};
  out->specific_energy = energy(s.r, s.v);
}

//------------------------------------------------------------------------------
/// \brief Destroy propagator and free memory.
///
/// \param handle  Opaque handle from numerov_orbit_create().
///   Handle is invalid after this call.
//------------------------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
void numerov_orbit_destroy(NumerovOrbitHandle* handle)
{
  delete handle;
}

//------------------------------------------------------------------------------
/// \brief Get physical constants (Earth).
///
/// \param out  Pointer to double[4] to receive: [mu, j2, r_earth, period_at_400km].
//------------------------------------------------------------------------------
EMSCRIPTEN_KEEPALIVE
void numerov_orbit_get_earth_constants(double* out)
{
  if (!out) return;
  constexpr double MU_EARTH = 3.986004418e14;   // m³/s²
  constexpr double J2 = 1.08263e-3;              // dimensionless
  constexpr double R_EARTH = 6.378137e6;       // m
  constexpr double ALT_400KM = 400e3;          // m
  constexpr double R0 = R_EARTH + ALT_400KM;
  const double PERIOD_400KM = 2.0 * 3.14159265358979323846 * std::sqrt(R0 * R0 * R0 / MU_EARTH);

  out[0] = MU_EARTH;
  out[1] = J2;
  out[2] = R_EARTH;
  out[3] = PERIOD_400KM;
}

} // extern "C"