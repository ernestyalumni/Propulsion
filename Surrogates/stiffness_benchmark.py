"""Measure what story 09 demands: the actual stiffness ratio of the chemistry
substep and the cost of the tuned classical integrator it would replace.

Definitions follow LaTeXandpdfs/SourceTermSurrogate.tex:
  state  phi = (T, Y_1..Y_ns) at constant pressure
  source S(phi) = d phi / dt from Cantera net production rates
  J = dS/dphi (central finite differences, relative perturbation)
  stiffness ratio  varsigma = max|Re lambda| / min_{nonconserved}|Re lambda|
The (n_e + 1) eigenvalues closest to zero are the conserved directions
(n_e element balances plus the enthalpy invariant) and are excluded from the
slow-mode denominator; their count is reported so the exclusion is auditable.
"""
from __future__ import annotations
import json, sys, time
import numpy as np
import cantera as ct

def source_term(gas: ct.Solution, T: float, Y: np.ndarray, p: float) -> np.ndarray:
    gas.TPY = T, p, Y
    rho = gas.density
    wdot = gas.net_production_rates * gas.molecular_weights   # kg/m^3/s
    dYdt = wdot / rho
    h_partial = gas.partial_molar_enthalpies / gas.molecular_weights  # J/kg
    dTdt = -np.dot(h_partial, wdot) / (rho * gas.cp_mass)
    return np.concatenate(([dTdt], dYdt))

def jacobian(gas, T, Y, p, rel=1e-6, abs_floor=1e-12):
    phi0 = np.concatenate(([T], Y))
    n = phi0.size
    J = np.empty((n, n))
    for j in range(n):
        h = rel * max(abs(phi0[j]), abs_floor)
        if j > 0:
            h = max(h, abs_floor)
        fp = phi0.copy(); fp[j] += h
        fm = phi0.copy(); fm[j] -= h
        Sp = source_term(gas, fp[0], np.clip(fp[1:], 0, None), p)
        Sm = source_term(gas, fm[0], np.clip(fm[1:], 0, None), p)
        J[:, j] = (Sp - Sm) / (2 * h)
    return J

def stiffness_along_trajectory(mech, fuel, T0, p, phi, n_samples=40, rtol=1e-8, atol=1e-15):
    gas = ct.Solution(mech)
    gas.set_equivalence_ratio(phi, fuel, "O2:1.0, N2:3.76")
    gas.TP = T0, p
    n_e = len([e for e in gas.element_names if any(gas.n_atoms(s, e) for s in gas.species_names)])
    r = ct.IdealGasConstPressureReactor(gas, clone=False)
    net = ct.ReactorNet([r]); net.rtol, net.atol = rtol, atol
    # first pass: find ignition delay (max dT/dt), then resample
    ts, Ts, Ys = [0.0], [r.T], [r.phase.Y.copy()]
    t_end = 0.05 if mech.startswith("h2") else 0.05
    while net.time < t_end:
        net.step(); ts.append(net.time); Ts.append(r.T); Ys.append(r.phase.Y.copy())
    ts, Ts = np.array(ts), np.array(Ts)
    dTdt = np.gradient(Ts, ts)
    i_ign = int(np.argmax(dTdt)); tau_ign = ts[i_ign]
    # sample states log-spaced from 1e-3 tau to 20 tau (covers induction, runaway, equilibration)
    t_samples = np.unique(np.concatenate([np.geomspace(1e-3*tau_ign, min(20*tau_ign, ts[-1]), n_samples), [tau_ign]]))
    rows = []
    for t in t_samples:
        i = int(np.searchsorted(ts, t)); i = min(i, len(ts)-1)
        J = jacobian(gas, Ts[i], Ys[i], p)
        lam = np.linalg.eigvals(J)
        re = np.abs(lam.real)
        order = np.sort(re)
        conserved = order[:n_e+1]
        active = order[n_e+1:]
        active = active[active > 0]
        fast, slow = active.max(), active.min()
        rows.append(dict(t=float(ts[i]), t_over_tau=float(ts[i]/tau_ign), T=float(Ts[i]),
                         fast=float(fast), slow=float(slow), ratio=float(fast/slow),
                         conserved_max=float(conserved.max()), n_conserved=int(n_e+1),
                         max_pos_real=float(lam.real.max())))
    worst = max(rows, key=lambda d: d["ratio"])
    at_ign = min(rows, key=lambda d: abs(d["t_over_tau"] - 1.0))
    median_ratio = float(np.median([d["ratio"] for d in rows]))
    return dict(at_ignition=at_ign, median_ratio=median_ratio, mech=mech, fuel=fuel, n_species=gas.n_species, n_reactions=gas.n_reactions,
                n_elements=n_e, T0=T0, p_atm=p/ct.one_atm, phi=phi, tau_ign_s=float(tau_ign),
                T_final=float(Ts[-1]), worst=worst, samples=rows)

def time_substep(mech, fuel, T0, p, phi, dt, rtol, atol, n_calls=None):
    """Cost of one chemistry substep advance(t+dt) per cell, sequential through ignition."""
    gas = ct.Solution(mech); gas.set_equivalence_ratio(phi, fuel, "O2:1.0, N2:3.76"); gas.TP = T0, p
    r = ct.IdealGasConstPressureReactor(gas, clone=False); net = ct.ReactorNet([r]); net.rtol, net.atol = rtol, atol
    # run through ~3x ignition delay
    tmp = stiffness_probe_tau(mech, fuel, T0, p, phi)
    t_end = 3 * tmp
    n = int(np.ceil(t_end / dt)) if n_calls is None else n_calls
    t = 0.0; t0 = time.perf_counter(); steps_before = 0
    for _ in range(n):
        t += dt; net.advance(t)
    wall = time.perf_counter() - t0
    return dict(dt=dt, rtol=rtol, atol=atol, n_calls=n, wall_s=wall,
                us_per_call=1e6*wall/n, calls_per_s=n/wall, t_end=t_end, T_end=float(r.T))

def time_substep_cold(mech, fuel, T0, p, phi, dt, rtol, atol, n_states=400):
    """Cold-start cost: states are sampled along the ignition trajectory, then for each
    state the reactor is reset and the integrator reinitialized before advance(dt),
    which is what an operator-split CFD solver does in every cell at every step."""
    gas = ct.Solution(mech); gas.set_equivalence_ratio(phi, fuel, "O2:1.0, N2:3.76"); gas.TP = T0, p
    r = ct.IdealGasConstPressureReactor(gas, clone=False); net = ct.ReactorNet([r]); net.rtol, net.atol = rtol, atol
    tau = stiffness_probe_tau(mech, fuel, T0, p, phi)
    states = []
    while net.time < 3 * tau:
        net.step(); states.append((r.T, r.phase.Y.copy()))
    idx = np.linspace(0, len(states) - 1, n_states).astype(int)
    states = [states[i] for i in idx]
    t0 = time.perf_counter()
    for T, Y in states:
        gas.TPY = T, p, Y
        r.syncState(); net.initial_time = 0.0; net.reinitialize()
        net.advance(dt)
    wall = time.perf_counter() - t0
    return dict(kind="cold", dt=dt, rtol=rtol, atol=atol, n_calls=len(states), wall_s=wall,
                us_per_call=1e6 * wall / len(states), calls_per_s=len(states) / wall)

_tau_cache = {}
def stiffness_probe_tau(mech, fuel, T0, p, phi):
    key = (mech, fuel, T0, p, phi)
    if key not in _tau_cache:
        gas = ct.Solution(mech); gas.set_equivalence_ratio(phi, fuel, "O2:1.0, N2:3.76"); gas.TP = T0, p
        r = ct.IdealGasConstPressureReactor(gas, clone=False); net = ct.ReactorNet([r])
        ts, Ts = [0.0], [r.T]
        while net.time < 0.05:
            net.step(); ts.append(net.time); Ts.append(r.T)
        ts, Ts = np.array(ts), np.array(Ts)
        _tau_cache[key] = float(ts[int(np.argmax(np.gradient(Ts, ts)))])
    return _tau_cache[key]

if __name__ == "__main__":
    out = dict(cantera=ct.__version__, numpy=np.__version__, stiffness=[], timing=[])
    cases = [
        ("h2o2.yaml", "H2", 1000.0, 1*ct.one_atm, 1.0),
        ("h2o2.yaml", "H2", 1200.0, 1*ct.one_atm, 1.0),
        ("h2o2.yaml", "H2", 1500.0, 1*ct.one_atm, 1.0),
        ("h2o2.yaml", "H2", 1200.0, 1*ct.one_atm, 0.5),
        ("h2o2.yaml", "H2", 1200.0, 1*ct.one_atm, 2.0),
        ("h2o2.yaml", "H2", 1200.0, 10*ct.one_atm, 1.0),
        ("gri30.yaml", "CH4", 1400.0, 1*ct.one_atm, 1.0),
        ("gri30.yaml", "CH4", 1400.0, 10*ct.one_atm, 1.0),
    ]
    for mech, fuel, T0, p, phi in cases:
        res = stiffness_along_trajectory(mech, fuel, T0, p, phi)
        w = res["worst"]
        a = res["at_ignition"]
        print(f"{mech:11s} {fuel:3s} T0={T0:6.0f} p={p/ct.one_atm:4.0f}atm phi={phi:3.1f} ns={res['n_species']:2d} "
              f"tau_ign={res['tau_ign_s']:.3e}s  worst ratio={w['ratio']:.1e} (t/tau={w['t_over_tau']:.2f})  "
              f"at-ignition ratio={a['ratio']:.1e} (fast={a['fast']:.1e}, slow={a['slow']:.1e} 1/s, T={a['T']:.0f}K)  median={res['median_ratio']:.1e}")
        out["stiffness"].append(res)
    print()
    for mech, fuel, T0, p, phi in [("h2o2.yaml","H2",1200.0,1*ct.one_atm,1.0), ("gri30.yaml","CH4",1400.0,1*ct.one_atm,1.0)]:
        for dt in (1e-6, 1e-7):
            for rtol, atol in ((1e-6, 1e-12), (1e-8, 1e-15)):
                tm = time_substep(mech, fuel, T0, p, phi, dt, rtol, atol)
                tm.update(mech=mech, fuel=fuel, T0=T0, p_atm=p/ct.one_atm, phi=phi)
                print(f"{mech:11s} dt={dt:.0e} rtol={rtol:.0e} atol={atol:.0e}  calls={tm['n_calls']:6d}  "
                      f"{tm['us_per_call']:8.1f} us/call  ({tm['calls_per_s']:9.0f} calls/s)  T_end={tm['T_end']:.0f}K")
                out["timing"].append(tm)
    print()
    for mech, fuel, T0, p, phi in [("h2o2.yaml","H2",1200.0,1*ct.one_atm,1.0), ("gri30.yaml","CH4",1400.0,1*ct.one_atm,1.0)]:
        for dt in (1e-6, 1e-7):
            for rtol, atol in ((1e-6, 1e-12), (1e-8, 1e-15)):
                tm = time_substep_cold(mech, fuel, T0, p, phi, dt, rtol, atol)
                tm.update(mech=mech, fuel=fuel, T0=T0, p_atm=p/ct.one_atm, phi=phi)
                print(f"COLD {mech:11s} dt={dt:.0e} rtol={rtol:.0e} atol={atol:.0e}  calls={tm['n_calls']:6d}  "
                      f"{tm['us_per_call']:8.1f} us/call  ({tm['calls_per_s']:9.0f} calls/s)")
                out["timing"].append(tm)
    for t in out["timing"]:
        t.setdefault("kind", "warm")
    json.dump(out, open("results/stiffness_benchmark.json", "w"), indent=1)
    print("\nwrote results/stiffness_benchmark.json")
