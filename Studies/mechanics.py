"""Independent teaching implementations; SI inputs unless nondimensionalized.

Derivations and book locators: ReadingRoom/resources.json, topic IDs
oscillator, rigid-body, nozzle. No Numerical Recipes distribution code is used.
"""
import math


def verlet(q, v, omega, h):
    """Velocity Verlet for q'' = -omega**2 q; fixed step, unit mass."""
    half = v - h * omega**2 * q / 2
    q += h * half
    return q, half - h * omega**2 * q / 2


def rk4(rhs, y, h):
    """Classical four-stage method derived from its order conditions."""
    a = rhs(y)
    b = rhs([x + h*k/2 for x, k in zip(y, a)])
    c = rhs([x + h*k/2 for x, k in zip(y, b)])
    d = rhs([x + h*k for x, k in zip(y, c)])
    return [x + h*(aa + 2*bb + 2*cc + dd)/6 for x, aa, bb, cc, dd in zip(y, a, b, c, d)]


def rigid_rhs(inertia):
    """y=(body omega, row-major R); R maps body vectors to inertial vectors."""
    i, j, k = inertia
    if min(inertia) <= 0:
        raise ValueError('Principal moments must be positive')
    def rhs(y):
        a, b, c = y[:3]
        spin = [(j-k)*b*c/i, (k-i)*c*a/j, (i-j)*a*b/k]
        hat = [[0, -c, b], [c, 0, -a], [-b, a, 0]]
        return spin + [sum(y[3+3*r+l]*hat[l][s] for l in range(3)) for r in range(3) for s in range(3)]
    return rhs


def area_mach(mach, gamma):
    """A/A* for steady, isentropic, quasi-1D calorically perfect gas flow."""
    if not math.isfinite(mach) or not math.isfinite(gamma) or mach <= 0 or gamma <= 1:
        raise ValueError('Require finite M > 0 and gamma > 1')
    return (2/(gamma+1)*(1+(gamma-1)*mach**2/2))**((gamma+1)/(2*(gamma-1)))/mach


def nozzle_mach(area, gamma, branch):
    """Bracket the selected physical branch; A/A* >= 1, gamma > 1."""
    if not math.isfinite(area) or area < 1 or not math.isfinite(gamma) or gamma <= 1:
        raise ValueError('Require finite A/A* >= 1 and gamma > 1')
    if branch not in ('subsonic', 'supersonic'):
        raise ValueError('Choose subsonic or supersonic')
    if area == 1:
        return 1.0
    lo, hi = (1e-12, 1.0) if branch == 'subsonic' else (1.0, 2.0)
    if branch == 'subsonic' and area_mach(lo, gamma) < area:
        raise ValueError('Area ratio exceeds supported subsonic bracket')
    if branch == 'supersonic':
        for _ in range(100):
            if area_mach(hi, gamma) >= area:
                break
            hi *= 2
        else:
            raise ValueError('Could not bracket the supersonic branch')
    for _ in range(100):
        mid = (lo+hi)/2
        f = area_mach(mid, gamma)-area
        if (f > 0) == (branch == 'supersonic'):
            hi = mid
        else:
            lo = mid
    return (lo+hi)/2


def numerical_checks():
    """Analytic oracles, convergence, conservation and branch checks."""
    errors = []
    for n in (80, 160, 320):
        q, v = 1.0, 0.0
        for _ in range(n):
            q, v = verlet(q, v, 1.0, 4/n)
        errors.append(math.hypot(q-math.cos(4), v+math.sin(4)))
    ratios = [errors[i]/errors[i+1] for i in range(2)]
    assert all(3.8 < r < 4.2 for r in ratios), ratios
    q, v, drift = 1.0, 0.0, 0.0
    for _ in range(10000):
        q, v = verlet(q, v, 1, .05)
        drift = max(drift, abs(q*q+v*v-1))
    assert drift < .001, drift
    oscillator = dict(errors=errors, refinement_ratios=ratios, max_relative_energy_drift=drift,
                      interval=4, steps=[80,160,320], long_run_steps=10000, long_run_step=.05)

    inertia = [2, 2, 3]
    initial = [.4, .2, .7, 1,0,0, 0,1,0, 0,0,1]
    energy0 = sum(i*w*w/2 for i,w in zip(inertia, initial))
    momentum0 = [i*w for i,w in zip(inertia, initial)]
    spin_errors = []
    for n in (200, 400):
        y = initial[:]
        for _ in range(n):
            y = rk4(rigid_rhs(inertia), y, 10/n)
        angle = .35*10
        exact = [.4*math.cos(angle)-.2*math.sin(angle), .4*math.sin(angle)+.2*math.cos(angle), .7]
        spin_errors.append(math.dist(y[:3], exact))
    assert 14 < spin_errors[0]/spin_errors[1] < 18
    r = [y[3+3*i:6+3*i] for i in range(3)]
    orthogonality = max(abs(sum(r[k][i]*r[k][j] for k in range(3))-(i==j)) for i in range(3) for j in range(3))
    determinant = sum(r[0][j]*(r[1][(j+1)%3]*r[2][(j+2)%3]-r[1][(j+2)%3]*r[2][(j+1)%3]) for j in range(3))
    energy = sum(i*w*w/2 for i,w in zip(inertia, y))
    momentum = [sum(r[a][b]*inertia[b]*y[b] for b in range(3)) for a in range(3)]
    assert orthogonality < 1e-7 and abs(determinant-1) < 1e-7
    assert abs(energy-energy0) < 1e-8 and math.dist(momentum, momentum0) < 1e-7
    rigid = dict(spin_errors=spin_errors, refinement_ratio=spin_errors[0]/spin_errors[1],
                 orthogonality_error=orthogonality, determinant=determinant,
                 energy_error=abs(energy-energy0), inertial_momentum_error=math.dist(momentum,momentum0),
                 inertia=inertia, initial_omega=initial[:3], interval=10, steps=[200,400])

    assert abs(area_mach(2, 1.4)-27/16) < 1e-14
    branches = [nozzle_mach(2.5, 1.4, branch) for branch in ('subsonic','supersonic')]
    assert branches[0] < 1 < branches[1]
    # Independent mass flux from stagnation relations, normalized by sonic flux.
    flux = [2.5*m/(1+.2*m*m)**3 * 1.2**3 for m in branches]
    assert max(abs(f-1) for f in flux) < 1e-12
    assert nozzle_mach(1, 1.4, 'subsonic') == nozzle_mach(1, 1.4, 'supersonic') == 1
    for a,g,b in ((.9,1.4,'subsonic'),(2,1,'supersonic'),(2,1.4,'unknown')):
        try:
            nozzle_mach(a,g,b)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid physical inputs accepted')
    return {'oscillator': oscillator, 'rigid-body': rigid,
            'nozzle': dict(area_ratio=2.5, gamma=1.4, mach_numbers=branches, normalized_mass_flux=flux)}
