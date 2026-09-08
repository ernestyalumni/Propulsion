"""Run with sage -python; exact symbolic checks, no SymPy-only substitute."""
import json
from sage.all import SR, matrix, vector, var, diff, QQ, latex, version


def checks():
    h, w, q, v = var('h w q v')
    a = matrix(SR, [[1-h*h*w*w/2, h], [-h*w*w*(1-h*h*w*w/4), 1-h*h*w*w/2]])
    metric = matrix(SR, [[w*w*(1-h*h*w*w/4), 0], [0, 1]])
    assert (a.det()-1).expand() == 0
    assert all(x.expand() == 0 for x in (a.transpose()*metric*a-metric).list())
    i,j,k,x,y,z = var('I_1 I_2 I_3 omega_1 omega_2 omega_3')
    spin = vector(SR, [(j-k)*y*z/i, (k-i)*z*x/j, (i-j)*x*y/k])
    energy = (i*x*x+j*y*y+k*z*z)/2
    norm = i*i*x*x+j*j*y*y+k*k*z*z
    assert sum(diff(energy,s)*ds for s,ds in zip((x,y,z),spin)).simplify_full() == 0
    assert sum(diff(norm,s)*ds for s,ds in zip((x,y,z),spin)).simplify_full() == 0
    m = var('M')
    area = (m*m+5)**3/(216*m)  # gamma = 7/5 exactly
    assert area.subs(M=1) == 1 and area.subs(M=2) == QQ(27)/16
    assert (diff(area,m)-5*(m*m-1)*(m*m+5)**2/(216*m*m)).simplify_full() == 0
    return {'schema': 1, 'sage_version': version(), 'topics': {
        'oscillator': {'passed': True, 'checks': ['symplectic determinant', 'exact modified quadratic invariant'], 'latex': latex(a)},
        'rigid-body': {'passed': True, 'checks': ['energy derivative zero', 'angular momentum norm derivative zero'], 'latex': latex(spin)},
        'nozzle': {'passed': True, 'checks': ['sonic limit', 'exact Mach-2 area', 'derivative and branch sign'], 'latex': latex(area)}}}


if __name__ == '__main__':
    print(json.dumps(checks()))
