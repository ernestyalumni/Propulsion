#!/usr/bin/env python3
"""
cosmos_orbit_demo.py — Cosmos GNC Orbital Animation
====================================================
G1 Autobot Cosmos — the space-recon Autobot who transforms into a flying saucer.
Named after Cosmos because this module handles GNC / astrodynamics demos.

Integrates LEO, GTO, and Hohmann transfer orbits using scipy DOP853 (equivalent
to DOPRI8 Dormand-Prince 8th-order), then animates them in a multi-panel 3D
matplotlib figure with a dark space aesthetic.

Panels:
  1 (main, large)  — 3D animated orbit: Earth sphere, satellite dots, path lines
  2                — Specific orbital energy error (relative, log scale)
  3                — Angular momentum error (relative, log scale)
  4                — Altitude vs time [km]

Usage:
  python cosmos_orbit_demo.py        # runs animation
  python cosmos_orbit_demo.py --save # save static snapshot only

Requirements: numpy, matplotlib, scipy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

try:
    from scipy.integrate import solve_ivp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not found — using simple RK4 fallback integrator", file=sys.stderr)

# ── Physical constants ────────────────────────────────────────────────────────

MU    = 3.986004418e14   # Earth gravitational parameter [m³/s²]
R_E   = 6.3781e6         # Earth equatorial radius [m]
PI    = np.pi
TWO_PI = 2.0 * PI

# ── Two-body equation of motion ───────────────────────────────────────────────

def two_body_eom(t, y):
    """dy/dt = f(t, y) for two-body Keplerian motion.

    y = [x, y, z, vx, vy, vz]  (SI units: m, m/s)
    """
    r3 = (y[0]**2 + y[1]**2 + y[2]**2) ** 1.5
    c  = -MU / r3
    return [y[3], y[4], y[5], c*y[0], c*y[1], c*y[2]]

# ── Keplerian elements → Cartesian state ──────────────────────────────────────

def elements_to_state(a, e, inc, raan=0.0, aop=0.0, nu=0.0):
    """Convert Keplerian elements to ECI Cartesian state vector."""
    p    = a * (1.0 - e**2)
    r    = p / (1.0 + e * np.cos(nu))
    sqmu = np.sqrt(MU / p)

    # Perifocal frame
    rP = r * np.cos(nu);  rQ = r * np.sin(nu)
    vP = -sqmu * np.sin(nu);  vQ = sqmu * (e + np.cos(nu))

    # Rotation matrix: R3(-RAAN) · R1(-i) · R3(-aop)
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc),  np.sin(inc)
    co, so = np.cos(aop),  np.sin(aop)

    Px =  cO*co - sO*so*ci;  Py =  sO*co + cO*so*ci;  Pz =  so*si
    Qx = -cO*so - sO*co*ci;  Qy = -sO*so + cO*co*ci;  Qz =  co*si

    x  = Px*rP + Qx*rQ;  y  = Py*rP + Qy*rQ;  z  = Pz*rP + Qz*rQ
    vx = Px*vP + Qx*vQ;  vy = Py*vP + Qy*vQ;  vz = Pz*vP + Qz*vQ

    return np.array([x, y, z, vx, vy, vz])

# ── Orbital period ────────────────────────────────────────────────────────────

def orbital_period(a):
    return TWO_PI * np.sqrt(a**3 / MU)

# ── Energy and angular momentum ───────────────────────────────────────────────

def specific_energy(y):
    r  = np.sqrt(y[0]**2 + y[1]**2 + y[2]**2)
    v2 = y[3]**2 + y[4]**2 + y[5]**2
    return 0.5 * v2 - MU / r

def angular_momentum_mag(y):
    h = np.cross(y[:3], y[3:])
    return np.linalg.norm(h)

def altitude_km(y):
    r = np.sqrt(y[0]**2 + y[1]**2 + y[2]**2)
    return (r - R_E) / 1e3

# ── Integrator ────────────────────────────────────────────────────────────────

def integrate_orbit(y0, t_end, t_eval_count=500):
    """Integrate the two-body EOM from t=0 to t=t_end.

    Returns (t, states) where states.shape = (6, len(t)).
    Uses DOP853 if scipy is available, else a simple fixed-step RK4.
    """
    t_span = (0.0, t_end)
    t_eval = np.linspace(0.0, t_end, t_eval_count)

    if HAS_SCIPY:
        sol = solve_ivp(
            two_body_eom,
            t_span,
            y0,
            method='DOP853',
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-10,
            dense_output=False,
        )
        return sol.t, sol.y        # sol.y shape: (6, n_points)

    # Fallback: simple RK4
    dt = t_end / (t_eval_count - 1)
    t_arr = np.zeros(t_eval_count)
    y_arr = np.zeros((6, t_eval_count))
    y_arr[:, 0] = y0
    y = y0.copy()
    for i in range(1, t_eval_count):
        t = (i - 1) * dt
        k1 = np.array(two_body_eom(t,          y))
        k2 = np.array(two_body_eom(t + dt/2,   y + dt/2 * k1))
        k3 = np.array(two_body_eom(t + dt/2,   y + dt/2 * k2))
        k4 = np.array(two_body_eom(t + dt,     y + dt   * k3))
        y = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t_arr[i]    = i * dt
        y_arr[:, i] = y
    return t_arr, y_arr

# ── Earth sphere ──────────────────────────────────────────────────────────────

def make_earth_sphere(ax, n=60):
    u = np.linspace(0, TWO_PI, n)
    v = np.linspace(0, PI, n)
    xs = R_E * np.outer(np.cos(u), np.sin(v))
    ys = R_E * np.outer(np.sin(u), np.sin(v))
    zs = R_E * np.outer(np.ones_like(u), np.cos(v))

    earth = ax.plot_surface(xs, ys, zs, color='#1a4080', alpha=0.85,
                            linewidth=0, antialiased=True, zorder=0)
    # Atmosphere halo
    atmo_r = R_E * 1.04
    xs2 = atmo_r * np.outer(np.cos(u), np.sin(v))
    ys2 = atmo_r * np.outer(np.sin(u), np.sin(v))
    zs2 = atmo_r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs2, ys2, zs2, color='#4488ff', alpha=0.08,
                    linewidth=0, antialiased=True, zorder=1)
    return earth

# ── Orbit configuration ───────────────────────────────────────────────────────

# LEO — ISS-like
a_leo  = 6.778e6;  e_leo = 0.001;  i_leo = np.radians(51.6)
T_leo  = orbital_period(a_leo)
y0_leo = elements_to_state(a_leo, e_leo, i_leo)

# GTO
a_gto  = 24500.0e3;  e_gto = 0.73
T_gto  = orbital_period(a_gto)
y0_gto = elements_to_state(a_gto, e_gto, 0.0)

# Hohmann transfer components
r_leo       = a_leo
r_geo       = 42164.0e3
a_transfer  = 0.5 * (r_leo + r_geo)
T_transfer  = PI * np.sqrt(a_transfer**3 / MU)  # half-period
T_geo       = orbital_period(r_geo)

v_leo          = np.sqrt(MU / r_leo)
v_transfer_p   = np.sqrt(MU * (2.0/r_leo - 1.0/a_transfer))
dv1            = v_transfer_p - v_leo

v_transfer_a   = np.sqrt(MU * (2.0/r_geo - 1.0/a_transfer))
v_geo          = np.sqrt(MU / r_geo)
dv2            = v_geo - v_transfer_a

# Parking LEO (circular, equatorial for clarity in Hohmann demo)
y0_park = elements_to_state(r_leo, 0.0, 0.0)

print("Integrating orbits (this may take a moment)...")
print(f"  LEO: T={T_leo/60:.1f} min, alt={( a_leo - R_E)/1e3:.0f} km")
print(f"  GTO: T={T_gto/3600:.2f} hr, e={e_gto}")
print(f"  Hohmann: dv1={dv1/1e3:.3f} km/s, dv2={dv2/1e3:.3f} km/s, "
      f"transfer={T_transfer/3600:.2f} hr")

NPTS = 600

t_leo, y_leo = integrate_orbit(y0_leo, T_leo, NPTS)
t_gto, y_gto = integrate_orbit(y0_gto, T_gto, NPTS)

# Hohmann: 3 segments
y0_h1 = y0_park.copy()
t_h1, y_h1 = integrate_orbit(y0_h1, T_leo, 200)

y0_h2 = y_h1[:, -1].copy()
y0_h2[4] += dv1   # dv1 applied at periapsis (purely in +y direction)
t_h2, y_h2 = integrate_orbit(y0_h2, T_transfer, 300)

y0_h3 = y_h2[:, -1].copy()
v_mag_apo = np.linalg.norm(y0_h3[3:])
y0_h3[3:] *= v_geo / v_mag_apo   # circularise at apoapsis
t_h3, y_h3 = integrate_orbit(y0_h3, T_geo, 300)

# Concatenate Hohmann
t_hoh = np.concatenate([t_h1, t_h1[-1] + t_h2, t_h1[-1] + t_h2[-1] + t_h3])
y_hoh = np.concatenate([y_h1, y_h2, y_h3], axis=1)

print("  Integration complete.")

# ── Compute diagnostics ───────────────────────────────────────────────────────

def diag_errors(t, y):
    eps0 = specific_energy(y[:, 0])
    h0   = angular_momentum_mag(y[:, 0])
    eps  = np.array([specific_energy(y[:, i]) for i in range(y.shape[1])])
    hm   = np.array([angular_momentum_mag(y[:, i]) for i in range(y.shape[1])])
    de   = np.abs((eps - eps0) / eps0)
    dh   = np.abs((hm - h0)   / h0)
    alt  = np.array([altitude_km(y[:, i]) for i in range(y.shape[1])])
    return de, dh, alt

de_leo, dh_leo, alt_leo = diag_errors(t_leo, y_leo)
de_gto, dh_gto, alt_gto = diag_errors(t_gto, y_gto)
de_hoh, dh_hoh, alt_hoh = diag_errors(t_hoh, y_hoh)

# ── Plot setup ────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    'figure.facecolor': '#050a18',
    'axes.facecolor':   '#050a18',
    'axes.edgecolor':   '#334466',
    'text.color':       '#c8d8ff',
    'axes.labelcolor':  '#88aaff',
    'xtick.color':      '#556688',
    'ytick.color':      '#556688',
    'grid.color':       '#1a2644',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
})

fig = plt.figure(figsize=(18, 10), facecolor='#050a18')
fig.suptitle(
    'Cosmos GNC — DOPRI5 Orbital Propagation',
    color='#88ccff', fontsize=16, fontweight='bold', y=0.97,
    fontfamily='monospace'
)

# Subtitle
fig.text(0.5, 0.935,
         'G1 Autobot Cosmos  ·  DOP853 Adaptive Integration  ·  LEO + GTO + Hohmann Transfer',
         ha='center', color='#5577aa', fontsize=9, fontfamily='monospace')

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    left=0.04, right=0.97,
    bottom=0.07, top=0.90,
    hspace=0.45, wspace=0.35
)

# Panel 1: 3D orbit (large, spans all 3 rows on left 2 columns)
ax3d = fig.add_subplot(gs[:, :2], projection='3d')
ax3d.set_facecolor('#050a18')

# Panel 2: Energy error
ax_e = fig.add_subplot(gs[0, 2])
ax_e.set_facecolor('#050a18')

# Panel 3: Angular momentum error
ax_h = fig.add_subplot(gs[1, 2])
ax_h.set_facecolor('#050a18')

# Panel 4: Altitude vs time
ax_a = fig.add_subplot(gs[2, 2])
ax_a.set_facecolor('#050a18')

# ── 3D orbital paths ──────────────────────────────────────────────────────────

km = 1e-3   # m → km scale factor for plotting (keep axes in km)

line_leo, = ax3d.plot(y_leo[0]*km, y_leo[1]*km, y_leo[2]*km,
                       color='#00eeff', lw=1.2, alpha=0.8, label='LEO (ISS)')
line_gto, = ax3d.plot(y_gto[0]*km, y_gto[1]*km, y_gto[2]*km,
                       color='#ff8c00', lw=1.0, alpha=0.7, label='GTO')
line_hoh, = ax3d.plot(y_hoh[0]*km, y_hoh[1]*km, y_hoh[2]*km,
                       color='#ffee44', lw=1.0, alpha=0.7, linestyle='--',
                       label='Hohmann')

# GEO reference circle
theta_geo = np.linspace(0, TWO_PI, 300)
gx = r_geo * np.cos(theta_geo) * km
gy = r_geo * np.sin(theta_geo) * km
ax3d.plot(gx, gy, np.zeros_like(gx),
          color='#cc44ff', lw=0.8, alpha=0.5, linestyle=':', label='GEO ring')

# Earth sphere
make_earth_sphere(ax3d, n=40)

# Satellite dots (animated)
dot_leo, = ax3d.plot([], [], [], 'o', color='#00ffff', ms=8, zorder=10,
                      markeredgecolor='white', markeredgewidth=0.5)
dot_gto, = ax3d.plot([], [], [], 'o', color='#ffaa22', ms=8, zorder=10,
                      markeredgecolor='white', markeredgewidth=0.5)
dot_hoh, = ax3d.plot([], [], [], 's', color='#ffee44', ms=7, zorder=10,
                      markeredgecolor='white', markeredgewidth=0.5)

# 3D axis styling
r_max = (r_geo + 2000e3) * km
ax3d.set_xlim(-r_max, r_max)
ax3d.set_ylim(-r_max, r_max)
ax3d.set_zlim(-r_max * 0.5, r_max * 0.5)
ax3d.set_xlabel('X [km]', labelpad=8)
ax3d.set_ylabel('Y [km]', labelpad=8)
ax3d.set_zlabel('Z [km]', labelpad=8)
ax3d.set_title('Orbital Trajectories — ECI Frame', color='#88ccff',
               fontsize=10, pad=8)
ax3d.tick_params(colors='#334466', labelsize=7)
ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor('#1a2644')
ax3d.yaxis.pane.set_edgecolor('#1a2644')
ax3d.zaxis.pane.set_edgecolor('#1a2644')
ax3d.grid(True, color='#1a2644', alpha=0.4)
ax3d.legend(loc='upper left', fontsize=7.5, framealpha=0.3,
            facecolor='#0a1428', edgecolor='#334466')
ax3d.view_init(elev=25, azim=45)

# ── Panel 2: Energy error ──────────────────────────────────────────────────────

ax_e.semilogy(t_leo / 60, np.maximum(de_leo, 1e-16),
              color='#00eeff', lw=1.2, label='LEO')
ax_e.semilogy(t_gto / 3600, np.maximum(de_gto, 1e-16),
              color='#ff8c00', lw=1.2, label='GTO')
ax_e.set_ylabel('|Δε/ε₀|', fontsize=8)
ax_e.set_title('Energy Error', color='#88ccff', fontsize=9, pad=4)
ax_e.set_xlabel('time [min / hr]', fontsize=7)
ax_e.legend(fontsize=7, framealpha=0.3, facecolor='#0a1428', edgecolor='#334466')
ax_e.grid(True, alpha=0.4)
ax_e.tick_params(labelsize=7)
ax_e.set_ylim(1e-14, 1e-4)

# ── Panel 3: Angular momentum error ───────────────────────────────────────────

ax_h.semilogy(t_leo / 60, np.maximum(dh_leo, 1e-16),
              color='#00eeff', lw=1.2, label='LEO')
ax_h.semilogy(t_gto / 3600, np.maximum(dh_gto, 1e-16),
              color='#ff8c00', lw=1.2, label='GTO')
ax_h.set_ylabel('|Δ|h|/|h₀||', fontsize=8)
ax_h.set_title('Angular Momentum Error', color='#88ccff', fontsize=9, pad=4)
ax_h.set_xlabel('time [min / hr]', fontsize=7)
ax_h.legend(fontsize=7, framealpha=0.3, facecolor='#0a1428', edgecolor='#334466')
ax_h.grid(True, alpha=0.4)
ax_h.tick_params(labelsize=7)
ax_h.set_ylim(1e-14, 1e-4)

# ── Panel 4: Altitude vs time ─────────────────────────────────────────────────

ax_a.plot(t_leo / 60, alt_leo,
          color='#00eeff', lw=1.2, label='LEO')
ax_a.plot(t_gto / 3600, alt_gto,
          color='#ff8c00', lw=1.2, label='GTO')
ax_a.plot(t_hoh / 3600, alt_hoh,
          color='#ffee44', lw=1.0, linestyle='--', label='Hohmann')

# Mark GEO altitude
ax_a.axhline((r_geo - R_E)/1e3, color='#cc44ff', lw=0.8, linestyle=':',
              alpha=0.6, label='GEO alt')

ax_a.set_ylabel('Altitude [km]', fontsize=8)
ax_a.set_title('Altitude vs Time', color='#88ccff', fontsize=9, pad=4)
ax_a.set_xlabel('time [min / hr]', fontsize=7)
ax_a.legend(fontsize=7, framealpha=0.3, facecolor='#0a1428', edgecolor='#334466',
            loc='upper left')
ax_a.grid(True, alpha=0.4)
ax_a.tick_params(labelsize=7)

# Stats text box
stats_text = (
    f"LEO:  a={a_leo/1e3:.0f} km  e={e_leo:.3f}  T={T_leo/60:.1f} min\n"
    f"GTO:  a={a_gto/1e3:.0f} km  e={e_gto:.2f}  T={T_gto/3600:.2f} hr\n"
    f"Hohmann: dv={( dv1+dv2)/1e3:.3f} km/s  t={T_transfer/3600:.2f} hr"
)
fig.text(0.5, 0.01, stats_text, ha='center', color='#5577aa',
         fontsize=8, fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#080f22',
                   edgecolor='#334466', alpha=0.8))

# ── Animation ─────────────────────────────────────────────────────────────────

N_FRAMES = 200
azim_start = 45.0

def frame_to_idx(frame, arr_len):
    """Map animation frame [0, N_FRAMES) → array index."""
    return int((frame / N_FRAMES) * (arr_len - 1))

def animate(frame):
    # Slowly rotate camera
    ax3d.view_init(elev=20 + 10 * np.sin(frame * TWO_PI / N_FRAMES),
                   azim=azim_start + frame * 180.0 / N_FRAMES)

    i_leo = frame_to_idx(frame, y_leo.shape[1])
    i_gto = frame_to_idx(frame, y_gto.shape[1])
    i_hoh = frame_to_idx(frame, y_hoh.shape[1])

    dot_leo.set_data_3d([y_leo[0, i_leo]*km],
                         [y_leo[1, i_leo]*km],
                         [y_leo[2, i_leo]*km])
    dot_gto.set_data_3d([y_gto[0, i_gto]*km],
                         [y_gto[1, i_gto]*km],
                         [y_gto[2, i_gto]*km])
    dot_hoh.set_data_3d([y_hoh[0, i_hoh]*km],
                         [y_hoh[1, i_hoh]*km],
                         [y_hoh[2, i_hoh]*km])

    return dot_leo, dot_gto, dot_hoh

anim = FuncAnimation(
    fig, animate,
    frames=N_FRAMES,
    interval=60,         # ms per frame  (~16 fps)
    blit=False,
    repeat=True,
)

# ── Save static snapshot ──────────────────────────────────────────────────────

snapshot_path = Path(__file__).parent / 'cosmos_orbit.png'
print(f"\nSaving static snapshot → {snapshot_path}")
animate(30)   # set a nice initial frame
fig.savefig(snapshot_path, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"  Saved: {snapshot_path}")

# ── Show animation ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Cosmos GNC orbital animation')
    parser.add_argument('--save', action='store_true',
                        help='Save static snapshot only, do not show animation')
    args = parser.parse_args()

    if args.save:
        print("Static snapshot saved. Exiting.")
        return

    print("\nShowing animation (close window to exit)...")
    plt.show()

if __name__ == '__main__':
    main()
