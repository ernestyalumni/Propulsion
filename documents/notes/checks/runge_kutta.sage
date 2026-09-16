"""Exact checks supporting the handwritten Runge--Kutta chapter.

Run with SageMath: sage runge_kutta.sage
No package installation, network, or repository mutation is required.
The symbolic polynomial examples supplement the general derivation; they do
not prove order for every vector field or validate an orbital implementation.
"""
import json
from sage.env import SAGE_VERSION

checks = []
def checked(name, condition, detail):
    if not condition:
        raise AssertionError(name)
    checks.append({"name": name, "passed": True, "detail": detail})

# Exact, independently specified classical tableau and its eight order sums.
A = matrix(QQ, [[0,0,0,0], [1/2,0,0,0], [0,1/2,0,0], [0,0,1,0]])
b = vector(QQ, [1/6,1/3,1/3,1/6])
ones = vector(QQ, [1,1,1,1])
c = A*ones
c2 = vector(QQ, [z^2 for z in c])
c3 = vector(QQ, [z^3 for z in c])
d = A*c
actual = [b*ones, b*c, b*c2, b*d, b*c3,
          b*vector(QQ, [c[i]*d[i] for i in range(4)]), b*(A*c2), b*(A^2*c)]
expected = [1,1/2,1/3,1/6,1/4,1/8,1/12,1/24]
checked("eight RK4 order conditions", actual == expected, [str(z) for z in actual])
P = PolynomialRing(QQ, 'z')
z = P.gen()
R = P(1) + sum((b*(A^j)*ones)*z^(j+1) for j in range(4))
checked("RK4 stability polynomial", R == 1+z+z^2/2+z^3/6+z^4/24, str(R))
checked("four stages cannot match exponential at degree five", R[5] == 0,
        {"numerical": "0", "exact": "1/120"})

# A nonlinear, nonautonomous two-component field, with mixed partials.
Q = PolynomialRing(QQ, names=('t','u','v'))
t,u,v = Q.gens()
coords = [u,v]
f = vector(Q, [t^3+u*v+u^2+t*u*v+v^2, t^2*u+u^2*v+v^3+t*v+1])
def D(p):
    return p.derivative(t) + sum(f[j]*p.derivative(coords[j]) for j in range(2))
def DV(w):
    return vector(Q, [D(p) for p in w])
J = matrix(Q, [[f[r].derivative(coords[j]) for j in range(2)] for r in range(2)])
U = DV(f)
B = vector(Q, [f[r].derivative(t,2)
    + 2*sum(f[r].derivative(t).derivative(coords[j])*f[j] for j in range(2))
    + sum(f[r].derivative(coords[j]).derivative(coords[k])*f[j]*f[k]
          for j in range(2) for k in range(2)) for r in range(2)])
C = vector(Q, [f[r].derivative(t,3)
    + 3*sum(f[r].derivative(t,2).derivative(coords[j])*f[j] for j in range(2))
    + 3*sum(f[r].derivative(t).derivative(coords[j]).derivative(coords[k])*f[j]*f[k]
            for j in range(2) for k in range(2))
    + sum(f[r].derivative(coords[j]).derivative(coords[k]).derivative(coords[l])*f[j]*f[k]*f[l]
          for j in range(2) for k in range(2) for l in range(2)) for r in range(2)])
cross = vector(Q, [sum((f[r].derivative(t).derivative(coords[j])
    + sum(f[r].derivative(coords[j]).derivative(coords[k])*f[k] for k in range(2)))*U[j]
    for j in range(2)) for r in range(2)])
checked("third total derivative, both components", DV(U) == B+J*U,
        "Polynomial identity in t,u,v over QQ; includes time and mixed state derivatives.")
checked("fourth total derivative, both components", DV(DV(U)) == C+3*cross+J*B+J^2*U,
        "Independent repeated total differentiation equals the displayed component formula.")

# Compare the full step with the exact solution series, retaining symbolic t,u,v.
H = PowerSeriesRing(Q, 'h', default_prec=6)
h = H.gen()
initial = [H(u),H(v)]
def polynomial_field(time, state):
    return [H(p(time,state[0],state[1])).add_bigoh(5) for p in f]
stages = []
for i in range(4):
    state = [initial[r]+h*sum(A[i,j]*stages[j][r] for j in range(i)) for r in range(2)]
    stages.append(polynomial_field(H(t)+c[i]*h, state))
numeric = [initial[r]+h*sum(b[i]*stages[i][r] for i in range(4)) for r in range(2)]
exact = initial[:]
derivative = f
for order in range(1,5):
    exact = [exact[r]+h^order*derivative[r]/factorial(order) for r in range(2)]
    derivative = DV(derivative)
for r in range(2):
    checked("nonlinear nonautonomous Taylor match, component "+str(r+1),
            all((numeric[r]-exact[r])[j] == 0 for j in range(5)),
            "Coefficients h^0 through h^4 agree identically in t,u,v over QQ.")

# A finite numerical check against an analytic solution; no production solver is imported.
import math
def rhs(time,value):
    return value-time*time+1
def advance(time,value,step):
    k1=rhs(time,value)
    k2=rhs(time+step/2,value+step*k1/2)
    k3=rhs(time+step/2,value+step*k2/2)
    k4=rhs(time+step,value+step*k3)
    return value+step*(k1+2*k2+2*k3+k4)/6
errors=[]
for count in [8,16,32]:
    step=1.0/int(count)
    value=0.5
    for i in range(int(count)):
        value=advance(i*step,value,step)
    errors.append(abs(value-(4.0-math.e/2.0)))
ratios=[errors[i]/errors[i+1] for i in range(2)]
checked("finite-interval RK4 convergence", all(14.0 < ratio < 17.0 for ratio in ratios),
        {"ivp":"y'=y-t^2+1, y(0)=1/2", "interval":[0,1],
         "steps":[8,16,32], "absolute_errors":[float(x) for x in errors],
         "error_ratios":[float(x) for x in ratios], "ratio_acceptance":[14,17]})
print(json.dumps({"schema":1,"sage_version":SAGE_VERSION,"passed":True,
    "scope":"Exact tableau checks and symbolic identities on a specified polynomial vector field, plus one finite convergence experiment.",
    "checks":checks},indent=2,default=lambda value: int(value) if isinstance(value,Integer) else str(value)))
