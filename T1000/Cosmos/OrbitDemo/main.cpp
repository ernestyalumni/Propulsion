/// \file main.cpp
/// \brief Cosmos GNC — Orbital Propagator Demo
///
/// Demonstrates:
///   1. LEO circular orbit (ISS-like): a=6778 km, e=0.001, i=51.6°
///   2. GTO (Geosynchronous Transfer Orbit): a=24500 km, e=0.73
///   3. Hohmann transfer LEO → GEO:
///        - delta-v1 at LEO periapsis (impulsive burn, increases apoapsis to GEO)
///        - coast on transfer ellipse for half a period
///        - delta-v2 at GEO apoapsis (circularise into GEO)
///
/// Integration: DOPRI5 adaptive stepper (Dormand-Prince 1980), 5(4) embedded pair.
/// Output: CSV files with columns t,x,y,z,vx,vy,vz,energy,h_mag
///
/// Usage:  ./OrbitDemo
///         Outputs: leo_orbit.csv, gto_orbit.csv, hohmann_transfer.csv
//------------------------------------------------------------------------------

#include "../../Source/OrbitalMechanics/Constants.hpp"
#include "../../Source/OrbitalMechanics/StateVector.hpp"
#include "../../Source/OrbitalMechanics/TwoBody.hpp"
#include "../../Source/OrbitalMechanics/OrbitalElements.hpp"
#include "../../Source/OrbitalMechanics/Propagator.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace OrbitalMechanics;

// ── CSV writer ────────────────────────────────────────────────────────────────

void write_csv(const std::string& filename,
               const PropagatorResult& result,
               double t_offset = 0.0)
{
    std::ofstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "[ERROR] Cannot open " << filename << "\n";
        return;
    }

    f << std::setprecision(12);
    f << "t,x,y,z,vx,vy,vz,energy,h_mag\n";

    for (std::size_t i = 0; i < result.times.size(); ++i)
    {
        const auto& s = result.states[i];
        f << result.times[i] + t_offset << ","
          << s[0] << ","
          << s[1] << ","
          << s[2] << ","
          << s[3] << ","
          << s[4] << ","
          << s[5] << ","
          << result.energies[i] << ","
          << result.hmags[i]   << "\n";
    }

    std::cout << "  Wrote " << result.times.size() << " rows -> " << filename << "\n";
}

// ── Helper: concatenate two PropagatorResults ─────────────────────────────────

void append_result(PropagatorResult& dst, const PropagatorResult& src, double t_offset)
{
    for (std::size_t i = 0; i < src.times.size(); ++i)
    {
        dst.times.push_back(src.times[i] + t_offset);
        dst.states.push_back(src.states[i]);
        dst.energies.push_back(src.energies[i]);
        dst.hmags.push_back(src.hmags[i]);
    }
}

// ── Print section header ──────────────────────────────────────────────────────

void section(const std::string& title)
{
    std::cout << "\n" << std::string(60, '=') << "\n"
              << "  " << title << "\n"
              << std::string(60, '=') << "\n";
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main()
{
    const double mu = Constants::MU_EARTH;

    std::cout << std::setprecision(6) << std::fixed;
    std::cout << "\n";
    std::cout << "  ██████╗ ██████╗ ███████╗███╗   ███╗ ██████╗ ███████╗\n";
    std::cout << " ██╔════╝██╔═══██╗██╔════╝████╗ ████║██╔═══██╗██╔════╝\n";
    std::cout << " ██║     ██║   ██║███████╗██╔████╔██║██║   ██║███████╗\n";
    std::cout << " ██║     ██║   ██║╚════██║██║╚██╔╝██║██║   ██║╚════██║\n";
    std::cout << " ╚██████╗╚██████╔╝███████║██║ ╚═╝ ██║╚██████╔╝███████║\n";
    std::cout << "  ╚═════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝\n";
    std::cout << "\n  G1 Autobot Cosmos — GNC & Astrodynamics Demo\n";
    std::cout << "  DOPRI5 Adaptive RK5(4) | Dormand-Prince 1980\n\n";

    // ─────────────────────────────────────────────────────────────────────────
    // 1. LEO Orbit  (ISS-like)
    // ─────────────────────────────────────────────────────────────────────────
    section("LEO — ISS-like Circular Orbit");

    const double a_leo = 6.778e6;   // m  (altitude ~400 km)
    const double e_leo = 0.001;
    const double i_leo = 51.6 * Constants::DEG2RAD;

    KeplerianElements leo_el{a_leo, e_leo, i_leo, 0.0, 0.0, 0.0};
    StateVector6 y0_leo = elements_to_state(leo_el);

    const double T_leo = orbital_period(a_leo);
    std::cout << "  a       = " << a_leo / 1e3 << " km\n";
    std::cout << "  e       = " << e_leo << "\n";
    std::cout << "  i       = " << 51.6 << " deg\n";
    std::cout << "  Period  = " << T_leo / 60.0 << " min\n";
    std::cout << "  v_circ  = " << circular_speed(a_leo) / 1e3 << " km/s\n";
    std::cout << "  alt     = " << (a_leo - Constants::R_EARTH) / 1e3 << " km\n";
    std::cout << "\n  Propagating one full orbit...\n";

    PropagatorResult leo_result = propagate(y0_leo, 0.0, T_leo,
                                            mu, 1e-9, 1e-9);

    const double e0_leo = leo_result.energies.front();
    const double ef_leo = leo_result.energies.back();
    const double h0_leo = leo_result.hmags.front();
    const double hf_leo = leo_result.hmags.back();

    std::cout << "  Steps   = " << leo_result.n_steps << "\n";
    std::cout << "  Energy drift = " << std::abs((ef_leo - e0_leo) / e0_leo)
              << " (relative)\n";
    std::cout << "  |h| drift    = " << std::abs((hf_leo - h0_leo) / h0_leo)
              << " (relative)\n";

    write_csv("leo_orbit.csv", leo_result);

    // ─────────────────────────────────────────────────────────────────────────
    // 2. GTO — Geosynchronous Transfer Orbit
    // ─────────────────────────────────────────────────────────────────────────
    section("GTO — Geosynchronous Transfer Orbit");

    const double a_gto = 24500.0e3;  // m
    const double e_gto = 0.73;

    const double r_p_gto = a_gto * (1.0 - e_gto);   // periapsis radius
    const double r_a_gto = a_gto * (1.0 + e_gto);   // apoapsis radius

    KeplerianElements gto_el{a_gto, e_gto, 0.0, 0.0, 0.0, 0.0};
    StateVector6 y0_gto = elements_to_state(gto_el);

    const double T_gto = orbital_period(a_gto);
    std::cout << "  a       = " << a_gto / 1e3 << " km\n";
    std::cout << "  e       = " << e_gto << "\n";
    std::cout << "  r_p     = " << r_p_gto / 1e3 << " km  (alt "
              << (r_p_gto - Constants::R_EARTH) / 1e3 << " km)\n";
    std::cout << "  r_a     = " << r_a_gto / 1e3 << " km  (alt "
              << (r_a_gto - Constants::R_EARTH) / 1e3 << " km)\n";
    std::cout << "  Period  = " << T_gto / 3600.0 << " hr\n";
    std::cout << "\n  Propagating one full orbit...\n";

    PropagatorResult gto_result = propagate(y0_gto, 0.0, T_gto,
                                            mu, 1e-9, 1e-9);

    const double e0_gto = gto_result.energies.front();
    const double ef_gto = gto_result.energies.back();
    const double h0_gto = gto_result.hmags.front();
    const double hf_gto = gto_result.hmags.back();

    std::cout << "  Steps   = " << gto_result.n_steps << "\n";
    std::cout << "  Energy drift = " << std::abs((ef_gto - e0_gto) / e0_gto)
              << " (relative)\n";
    std::cout << "  |h| drift    = " << std::abs((hf_gto - h0_gto) / h0_gto)
              << " (relative)\n";

    write_csv("gto_orbit.csv", gto_result);

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Hohmann Transfer  LEO → GEO
    // ─────────────────────────────────────────────────────────────────────────
    section("Hohmann Transfer: LEO → GEO");

    // Orbit radii  (circular: r = a)
    const double r_leo = a_leo;                   // LEO radius
    const double r_geo = 42164.0e3;               // GEO radius (geostationary)

    // Transfer ellipse semi-major axis
    const double a_transfer = 0.5 * (r_leo + r_geo);

    // ── Delta-v calculations ──────────────────────────────────────────────────
    // v_leo = circular speed at LEO
    const double v_leo = std::sqrt(mu / r_leo);

    // v_transfer_p = speed at transfer ellipse periapsis (vis-viva)
    const double v_transfer_p = std::sqrt(mu * (2.0/r_leo - 1.0/a_transfer));

    // dv1 = burn at LEO periapsis to enter transfer ellipse
    const double dv1 = v_transfer_p - v_leo;

    // v_transfer_a = speed at transfer ellipse apoapsis
    const double v_transfer_a = std::sqrt(mu * (2.0/r_geo - 1.0/a_transfer));

    // v_geo = circular speed at GEO
    const double v_geo = std::sqrt(mu / r_geo);

    // dv2 = burn at GEO apoapsis to circularise
    const double dv2 = v_geo - v_transfer_a;

    // Total delta-v
    const double dv_total = dv1 + dv2;

    // Transfer time = half the transfer ellipse period
    const double T_transfer = Constants::PI * std::sqrt(
        a_transfer * a_transfer * a_transfer / mu);

    std::cout << "  r_LEO          = " << r_leo / 1e3 << " km\n";
    std::cout << "  r_GEO          = " << r_geo / 1e3 << " km\n";
    std::cout << "  a_transfer     = " << a_transfer / 1e3 << " km\n";
    std::cout << "\n";
    std::cout << "  v_LEO          = " << v_leo / 1e3 << " km/s\n";
    std::cout << "  v_transfer_p   = " << v_transfer_p / 1e3 << " km/s\n";
    std::cout << "  dv1 (LEO→XFR)  = " << dv1 / 1e3 << " km/s\n";
    std::cout << "\n";
    std::cout << "  v_transfer_a   = " << v_transfer_a / 1e3 << " km/s\n";
    std::cout << "  v_GEO          = " << v_geo / 1e3 << " km/s\n";
    std::cout << "  dv2 (XFR→GEO)  = " << dv2 / 1e3 << " km/s\n";
    std::cout << "\n";
    std::cout << "  dv_total       = " << dv_total / 1e3 << " km/s\n";
    std::cout << "  Transfer time  = " << T_transfer / 3600.0 << " hr\n\n";

    // ── Segment 1: LEO coast (one orbit, to show the parking orbit) ───────────
    std::cout << "  [1/3] LEO parking orbit...\n";
    StateVector6 y_leo_park = elements_to_state(
        KeplerianElements{r_leo, 0.0, 0.0, 0.0, 0.0, 0.0});
    PropagatorResult seg1 = propagate(y_leo_park, 0.0, T_leo,
                                      mu, 1e-9, 1e-9);

    // ── Segment 2: Apply dv1 and propagate transfer ellipse ───────────────────
    std::cout << "  [2/3] Transfer ellipse (dv1=" << dv1/1e3 << " km/s)...\n";

    // At LEO periapsis (ν=0), the velocity is purely tangential (+y in this setup)
    // Add dv1 in the prograde direction (tangential at periapsis = +y)
    StateVector6 y_xfr_start = seg1.states.back();
    y_xfr_start[4] += dv1;   // prograde tangential burn at periapsis

    PropagatorResult seg2 = propagate(y_xfr_start, 0.0, T_transfer,
                                      mu, 1e-9, 1e-9);

    // ── Segment 3: Apply dv2, coast a bit (show GEO) ─────────────────────────
    std::cout << "  [3/3] GEO insertion (dv2=" << dv2/1e3 << " km/s)...\n";

    StateVector6 y_geo_start = seg2.states.back();
    // At GEO apoapsis, velocity is again tangential; add dv2 prograde
    // Compute the actual velocity direction from the state
    const double v_mag_apo = speed(y_geo_start);
    y_geo_start[3] *= (v_geo / v_mag_apo);
    y_geo_start[4] *= (v_geo / v_mag_apo);
    y_geo_start[5] *= (v_geo / v_mag_apo);

    const double T_geo = orbital_period(r_geo);
    PropagatorResult seg3 = propagate(y_geo_start, 0.0, T_geo,
                                      mu, 1e-9, 1e-9);

    // ── Assemble combined trajectory ──────────────────────────────────────────
    PropagatorResult hohmann;
    hohmann.n_steps = seg1.n_steps + seg2.n_steps + seg3.n_steps;

    append_result(hohmann, seg1, 0.0);
    double t_seg2_start = seg1.times.back();
    append_result(hohmann, seg2, t_seg2_start);
    double t_seg3_start = t_seg2_start + seg2.times.back();
    append_result(hohmann, seg3, t_seg3_start);

    double t_total_hr = (t_seg3_start + seg3.times.back()) / 3600.0;
    std::cout << "\n  Total trajectory time = " << t_total_hr << " hr\n";
    std::cout << "  Total integration steps = " << hohmann.n_steps << "\n";

    // GEO verification
    const auto& s_geo_final = seg3.states.back();
    double r_geo_actual = radius(s_geo_final);
    double v_geo_actual = speed(s_geo_final);
    std::cout << "\n  GEO verification:\n";
    std::cout << "    r_actual = " << r_geo_actual / 1e3 << " km  (target "
              << r_geo / 1e3 << " km)\n";
    std::cout << "    v_actual = " << v_geo_actual / 1e3 << " km/s  (target "
              << v_geo / 1e3 << " km/s)\n";
    std::cout << "    GEO period = "
              << orbital_period(radius(seg3.states.front())) / 3600.0
              << " hr  (target 24.0 hr)\n";

    write_csv("hohmann_transfer.csv", hohmann);

    // ─────────────────────────────────────────────────────────────────────────
    // Summary
    // ─────────────────────────────────────────────────────────────────────────
    section("Summary");

    std::cout << "  Orbit          | Alt (km)     | Period       | Steps\n";
    std::cout << "  " << std::string(56, '-') << "\n";
    std::cout << "  LEO (ISS)      | "
              << std::setw(12) << (a_leo - Constants::R_EARTH)/1e3
              << " | " << std::setw(8) << T_leo/60.0 << " min    | "
              << leo_result.n_steps << "\n";
    std::cout << "  GTO            | "
              << std::setw(12) << (r_p_gto - Constants::R_EARTH)/1e3
              << " | " << std::setw(8) << T_gto/3600.0 << " hr     | "
              << gto_result.n_steps << "\n";
    std::cout << "  Hohmann (3seg) | "
              << std::setw(12) << (r_geo - Constants::R_EARTH)/1e3
              << " | " << std::setw(8) << T_geo/3600.0 << " hr     | "
              << hohmann.n_steps << "\n";

    std::cout << "\n  Delta-v budget (LEO → GEO Hohmann):\n";
    std::cout << "    dv1 = " << dv1 / 1e3 << " km/s  (LEO periapsis boost)\n";
    std::cout << "    dv2 = " << dv2 / 1e3 << " km/s  (GEO apoapsis circularise)\n";
    std::cout << "    dv_total = " << dv_total / 1e3 << " km/s\n";

    std::cout << "\n  Output files:\n";
    std::cout << "    leo_orbit.csv\n";
    std::cout << "    gto_orbit.csv\n";
    std::cout << "    hohmann_transfer.csv\n";

    std::cout << "\n[Cosmos GNC demo complete]\n\n";

    return 0;
}
