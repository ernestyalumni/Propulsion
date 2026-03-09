/**
 * @file NumerovOrbitDemo.cpp
 * @brief Astrodynamics demo: Numerov integration for orbital propagation
 *
 * References:
 * - Curtis, "Orbital Mechanics for Engineering Students", §4.7 (J2), §10
 * - Hintz, "Orbital Mechanics and Astrodynamics", Ch. 5
 * - Bate, "Fundamentals of Astrodynamics", §9.2-9.6
 *
 * Startup step (r0→r1 for Numerov init):
 *   Uses StepperDopr5::dy() — one fixed non-adaptive DOPRI5 step.
 *   This reuses Numerical::ODE::StepperDopr5 directly rather than rolling a
 *   custom RK4. dy() is called instead of step() to avoid adaptive stepsize
 *   control; we want exactly one step of size dt.
 *
 *   Note: StepperDopr5's inner loops compare std::size_t (loop var) against
 *   int n_ (StepperBase member). The pragma below suppresses the
 *   -Wsign-compare warning that fires on template instantiation.  The proper
 *   fix would be to change n_ / n_eqns_ to std::size_t in StepperBase.
 *
 * Build: mkdir -p build && cd build && cmake .. && make NumerovOrbitDemo
 * Run:   ./NumerovOrbitDemo
 */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// StepperDopr5 is fully header-only (all methods defined in the header as
// templates). No additional link target is required beyond the include path.
// StepperDopr5.h has two known issues when its templates are instantiated:
//   1. Inner loops: std::size_t i vs int n_  → -Wsign-compare
//   2. a72 (DOPRI5 zero coefficient, defined but unused) → -Wunused-variable
// Both are suppressed here. The proper fix is in StepperBase (n_ → std::size_t)
// and in dy() (mark a72 with [[maybe_unused]] or fold it into the formula).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "Numerical/ODE/StepperDopr5.h"
#pragma GCC diagnostic pop

using Numerical::ODE::StepperDopr5;

// ── Physical constants (SI) ──────────────────────────────────────────────────
constexpr double MU_EARTH = 3.986004418e14;  // m³/s²
constexpr double R_EARTH  = 6.378137e6;      // m
constexpr double J2       = 1.08263e-3;      // Earth oblateness

// ── 3D vector ────────────────────────────────────────────────────────────────
struct Vector3D
{
  double x, y, z;

  Vector3D(double x_ = 0, double y_ = 0, double z_ = 0)
    : x(x_), y(y_), z(z_) {}

  Vector3D operator+(const Vector3D& o) const { return {x+o.x, y+o.y, z+o.z}; }
  Vector3D operator-(const Vector3D& o) const { return {x-o.x, y-o.y, z-o.z}; }
  Vector3D operator*(double s)           const { return {x*s,   y*s,   z*s};   }
  Vector3D operator/(double s)           const { return {x/s,   y/s,   z/s};   }

  double norm()    const { return std::sqrt(x*x + y*y + z*z); }
  double norm_sq() const { return x*x + y*y + z*z; }
};

// ── Accelerations ────────────────────────────────────────────────────────────
Vector3D twoBodyAccel(const Vector3D& r)
{
  const double rn = r.norm();
  return r * (-MU_EARTH / (rn * rn * rn));
}

Vector3D j2Perturbation(const Vector3D& r)
{
  const double rn  = r.norm();
  const double rSq = rn * rn;
  const double zSqOvrRSq = (r.z * r.z) / rSq;
  const double factor =
    -1.5 * MU_EARTH * J2 * R_EARTH * R_EARTH / (rSq * rSq * rn);
  return Vector3D(
    factor * r.x * (1.0 - 5.0 * zSqOvrRSq),
    factor * r.y * (1.0 - 5.0 * zSqOvrRSq),
    factor * r.z * (3.0 - 5.0 * zSqOvrRSq));
}

Vector3D totalAccel(const Vector3D& r, bool use_j2)
{
  Vector3D a = twoBodyAccel(r);
  if (use_j2) a = a + j2Perturbation(r);
  return a;
}

// ── State ────────────────────────────────────────────────────────────────────
struct State
{
  double   t;
  Vector3D r, v;
};

// ── DOPRI5 startup step ──────────────────────────────────────────────────────
// Advances one fixed step of size h from s0 using Dormand-Prince RK5(4).
// Calls StepperDopr5::dy() directly — one non-adaptive step, O(h^5) LTE.
// The stepper's adaptive machinery (step() / Controller::success()) is NOT
// used; we want exactly r1 = r(t0 + h) as Numerov's second starting point.
//
// State layout: y = [rx, ry, rz, vx, vy, vz],  dydt = [v, a(r)]
State dopr5StartupStep(const State& s0, double h, bool use_j2)
{
  std::vector<double> y = {
    s0.r.x, s0.r.y, s0.r.z,
    s0.v.x, s0.v.y, s0.v.z
  };
  std::vector<double> dydx(6);
  double x = s0.t;

  // Derivative functor — signature matches dy()'s call pattern:
  //   derivatives(double t, vector<double>& yin, vector<double>& dout)
  // dy() passes lvalue vectors for both yin and dout, so the reference
  // parameters here bind correctly and dout receives the computed values.
  auto deriv = [use_j2](
    double       /*t*/,
    std::vector<double>& yin,
    std::vector<double>& dout)
  {
    const Vector3D r{yin[0], yin[1], yin[2]};
    const Vector3D a = totalAccel(r, use_j2);
    dout[0] = yin[3]; dout[1] = yin[4]; dout[2] = yin[5]; // dr/dt = v
    dout[3] = a.x;    dout[4] = a.y;    dout[5] = a.z;    // dv/dt = a(r)
  };

  // Evaluate initial derivative into dydx
  deriv(x, y, dydx);

  // Construct stepper — tolerance args are unused since we call dy(), not step()
  StepperDopr5<decltype(deriv)> stepper{y, dydx, x, 1e-12, 1e-12, false};

  // Single fixed DOPRI5 step: fills stepper.y_out_ with state at t0 + h
  stepper.dy(h, deriv);

  return State{
    s0.t + h,
    Vector3D{stepper.y_out_[0], stepper.y_out_[1], stepper.y_out_[2]},
    Vector3D{stepper.y_out_[3], stepper.y_out_[4], stepper.y_out_[5]}
  };
}

// ── Numerov propagator ───────────────────────────────────────────────────────
// Implements the Störmer δ-form (HNW §III.10, Eq. 10.38') to avoid the
// double-root rounding-error growth of the naive 3-point recurrence.
class NumerovOrbitPropagator
{
public:
  double              h;
  bool                use_j2;
  std::vector<Vector3D> r_hist;
  std::vector<Vector3D> a_hist;
  std::vector<double>   t_hist;

  NumerovOrbitPropagator(double step, bool j2)
    : h(step), use_j2(j2) {}

  void init(const State& s0, const State& s1)
  {
    r_hist.push_back(s0.r);
    r_hist.push_back(s1.r);
    a_hist.push_back(totalAccel(s0.r, use_j2));
    a_hist.push_back(totalAccel(s1.r, use_j2));
    t_hist.push_back(s0.t);
    t_hist.push_back(s1.t);
  }

  void step()
  {
    const std::size_t n   = r_hist.size() - 1;
    const Vector3D& r_n   = r_hist[n];
    const Vector3D& r_nm1 = r_hist[n-1];
    const Vector3D& a_n   = a_hist[n];
    const Vector3D& a_nm1 = a_hist[n-1];

    // PECECE: predict (explicit Störmer), evaluate, correct (Numerov)
    const Vector3D r_pred = r_n * 2.0 - r_nm1 + a_n * (h * h);
    const Vector3D a_pred = totalAccel(r_pred, use_j2);

    const Vector3D r_new =
      r_n * 2.0 - r_nm1
      + (a_pred + a_n * 10.0 + a_nm1) * (h * h / 12.0);
    const Vector3D a_new = totalAccel(r_new, use_j2);

    r_hist.push_back(r_new);
    a_hist.push_back(a_new);
    t_hist.push_back(t_hist.back() + h);
  }

  State getState() const
  {
    const std::size_t n = r_hist.size() - 1;
    // Störmer-Verlet integer-time velocity (StormerRule.tex §4, Eq. alg-vel)
    const Vector3D v = (r_hist[n] - r_hist[n-1]) / h;
    return State{t_hist[n], r_hist[n], v};
  }
};

// ── Orbital mechanics helpers ────────────────────────────────────────────────
double specificEnergy(const Vector3D& r, const Vector3D& v)
{
  return v.norm_sq() / 2.0 - MU_EARTH / r.norm();
}

// Analytical J2 nodal regression (rad/s) — Curtis Eq. 4.52
double j2NodalRegression(double a, double e, double i)
{
  const double n = std::sqrt(MU_EARTH / (a * a * a));
  const double p = a * (1.0 - e * e);
  return n * -1.5 * J2 * (R_EARTH / p) * (R_EARTH / p) * std::cos(i);
}

// Analytical J2 perigee advance (rad/s) — Curtis Eq. 4.53
double j2PerigeeAdvance(double a, double e, double i)
{
  const double n = std::sqrt(MU_EARTH / (a * a * a));
  const double p = a * (1.0 - e * e);
  return n * 0.75 * J2 * (R_EARTH / p) * (R_EARTH / p)
         * (5.0 * std::cos(i) * std::cos(i) - 1.0);
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main()
{
  std::cout
    << "========================================\n"
    << "Numerov Astrodynamics Demo\n"
    << "Cowell's method + Numerov integration\n"
    << "Startup: StepperDopr5::dy() (DOPRI5, fixed h)\n"
    << "========================================\n\n";

  const double altitude    = 400e3;
  const double r0          = R_EARTH + altitude;
  const double v_circ      = std::sqrt(MU_EARTH / r0);
  const double inclination = 28.5 * M_PI / 180.0;

  State s0;
  s0.t = 0.0;
  s0.r = Vector3D(r0, 0.0, 0.0);
  s0.v = Vector3D(0.0,
                  v_circ * std::cos(inclination),
                  v_circ * std::sin(inclination));

  const double period = 2.0 * M_PI * std::sqrt(r0 * r0 * r0 / MU_EARTH);

  std::cout << "Initial Conditions:\n"
            << "  Altitude:  " << altitude / 1000.0 << " km\n"
            << "  Incl.:     " << inclination * 180.0 / M_PI << " deg\n"
            << "  v_circ:    " << v_circ << " m/s\n"
            << "  Period:    " << period / 60.0 << " min\n\n";

  const double omega_dot = j2NodalRegression(r0, 0.0, inclination);
  const double argp_dot  = j2PerigeeAdvance(r0,  0.0, inclination);
  std::cout << "Analytical J2 (Curtis Eq. 4.52-4.53):\n"
            << "  RAAN drift:    "
            << omega_dot * 180.0 / M_PI * 86400.0 << " deg/day\n"
            << "  Perigee adv.:  "
            << argp_dot  * 180.0 / M_PI * 86400.0 << " deg/day\n\n";

  const double dt       = 60.0;
  const int    n_orbits = 10;
  const int    n_steps  = static_cast<int>(n_orbits * period / dt);

  std::cout << "Simulation: " << n_orbits << " orbits, dt=" << dt << " s\n\n";

  for (bool use_j2 : {false, true})
  {
    std::cout << "========================================\n"
              << (use_j2 ? "CASE 2: Two-body + J2" : "CASE 1: Pure Two-body")
              << "\n========================================\n";

    NumerovOrbitPropagator prop(dt, use_j2);

    // One DOPRI5 startup step — replaces the previously hand-rolled rk4Step
    const State s1 = dopr5StartupStep(s0, dt, use_j2);
    prop.init(s0, s1);

    const double E0 = specificEnergy(s0.r, s0.v);
    std::cout << "Initial energy: " << std::scientific << E0 << " J/kg\n";

    const std::string filename =
      use_j2 ? "numerov_j2.csv" : "numerov_twobody.csv";
    std::ofstream out(filename);
    out << std::scientific << std::setprecision(15);
    out << "t,orbit,r_x,r_y,r_z,energy,error\n";
    out << s0.t << ",0,"
        << s0.r.x << "," << s0.r.y << "," << s0.r.z << ","
        << E0 << ",0\n";

    double max_err  = 0.0;
    int    last_orbit = 0;

    for (int i = 2; i <= n_steps; ++i)
    {
      prop.step();

      const State  current = prop.getState();
      const double E       = specificEnergy(current.r, current.v);
      const double err     = std::abs((E - E0) / E0);
      if (err > max_err) max_err = err;

      const int orbit = static_cast<int>(current.t / period);

      if (i % 10 == 0)
      {
        out << current.t << "," << orbit << ","
            << current.r.x << "," << current.r.y << "," << current.r.z << ","
            << E << "," << err << "\n";
      }

      if (orbit != last_orbit)
      {
        std::cout << "  Orbit " << orbit
                  << ": r=" << std::fixed << std::setprecision(2)
                  << current.r.norm() / 1000.0 << " km"
                  << ", err=" << std::scientific << err << "\n";
        last_orbit = orbit;
      }
    }

    out.close();
    std::cout << "\nResults:\n"
              << "  Max energy error: " << std::scientific << max_err << "\n"
              << "  Output: " << filename << "\n\n";
  }

  std::cout << "========================================\n"
            << "Demo Complete!\n"
            << "========================================\n";
  return 0;
}
