#!/usr/bin/env python3
"""Independent Fraction oracle: derive weights using the eight explicit order
conditions through order four (HNW I, II.1), then compare the Rust CLI.
Runs offline, using temporary build products. No SymPy or Fortran needed.
"""
from fractions import Fraction as F
from math import factorial
from pathlib import Path
import subprocess
import tempfile


def dot(x, y):
    return sum((a*b for a, b in zip(x, y)), F(0))


def order_equations(a, p):
    """Explicit equations, deliberately independent of Rust's tree enumeration."""
    c = [sum(row, F(0)) for row in a]
    ac = [dot(row, c) for row in a]
    c2 = [x*x for x in c]
    columns = [[F(1)]*len(a), c, c2, ac, [x**3 for x in c],
               [x*y for x, y in zip(c, ac)],
               [dot(row, c2) for row in a], [dot(row, ac) for row in a]]
    targets = [F(1), F(1,2), F(1,3), F(1,6), F(1,4), F(1,8), F(1,12), F(1,24)]
    count = {1: 1, 2: 2, 3: 4, 4: 8}[p]
    return [col + [target] for col, target in zip(columns[:count], targets[:count])]


def derive_weights(a, p):
    """Unique solution of the overdetermined linear system for b, given A."""
    rows = order_equations(a, p)
    n = len(a)
    pivots = []
    for col in range(n):
        pivot = next((i for i in range(len(pivots), len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        k = len(pivots)
        rows[k], rows[pivot] = rows[pivot], rows[k]
        scale = rows[k][col]
        rows[k] = [v/scale for v in rows[k]]
        for i in range(len(rows)):
            if i != k:
                scale = rows[i][col]
                rows[i] = [v-scale*w for v,w in zip(rows[i], rows[k])]
        pivots.append(col)
    if any(not any(row[:n]) and row[n] for row in rows):
        raise ValueError('inconsistent order conditions')
    if len(pivots) != n:
        raise ValueError('nonunique output weights')
    result = [F(0)]*n
    for i, col in enumerate(pivots):
        result[col] = rows[i][n]
    return result


def fixtures():
    # Independently specified classical tableaux; Python strings avoid float conversion.
    raw = [
        ('euler',1,[[0]],[1]),
        ('midpoint',2,[[0,0],['1/2',0]],[0,1]),
        ('heun2',2,[[0,0],[1,0]],['1/2','1/2']),
        ('ralston2',2,[[0,0],['2/3',0]],['1/4','3/4']),
        ('kutta3',3,[[0,0,0],['1/2',0,0],[-1,2,0]],['1/6','2/3','1/6']),
        ('rk4',4,[[0,0,0,0],['1/2',0,0,0],[0,'1/2',0,0],[0,0,1,0]],['1/6','1/3','1/3','1/6']),
        ('rk38',4,[[0,0,0,0],['1/3',0,0,0],['-1/3',1,0,0],[1,-1,1,0]],['1/8','3/8','3/8','1/8']),
    ]
    return [(name,p,[[F(v) for v in row] for row in a],[F(v) for v in b]) for name,p,a,b in raw]


def main():
    source = Path(__file__).with_name('exact_rk.rs')
    with tempfile.TemporaryDirectory(prefix='propulsion-exact-rk-') as tmp:
        binary = str(Path(tmp)/'derive')
        tests = str(Path(tmp)/'tests')
        subprocess.run(['rustc','--edition=2021','--test',str(source),'-o',tests],check=True)
        subprocess.run([tests],check=True)
        subprocess.run(['rustc','--edition=2021','-O',str(source),'-o',binary],check=True)
        lines = subprocess.check_output([binary],text=True).splitlines()
    expected = []
    for name,p,a,known in fixtures():
        b = derive_weights(a,p)
        if b != known:
            raise AssertionError((name,b,known))
        for row in order_equations(a,p):
            if dot(row[:-1],b) != row[-1]:
                raise AssertionError((name,'failed order equation'))
        # Independently construct stability polynomial, retaining all stage powers.
        v = [F(1)]*len(a)
        stability = [F(1)]
        for _ in a:
            stability.append(dot(b,v))
            v = [dot(row,v) for row in a]
        if stability != [F(1,factorial(k)) for k in range(p+1)]:
            raise AssertionError((name,'stability polynomial'))
        c = [sum(row,F(0)) for row in a]
        encode = lambda values: ','.join(map(str,values))
        expected.append('\t'.join([name,str(p),encode(c),encode(b),encode(stability)]))
    if lines != expected:
        raise AssertionError((lines,expected))
    # This three-stage tableau matches exp(z) through degree three but fails
    # b*c^2=1/3: a concrete counterexample to linear-only order verification.
    a = [[F(0)]*3, [F(1),F(0),F(0)], [F(0),F(1),F(0)]]
    b = [F(1,2),F(1,3),F(1,6)]
    equations = order_equations(a,3)
    if not all(dot(equations[i][:-1],b)==equations[i][-1] for i in (0,1,3)):
        raise AssertionError('counterexample should pass the linear conditions')
    if dot(equations[2][:-1],b)==equations[2][-1]:
        raise AssertionError('nonlinear third-order condition must detect failure')
    try:
        derive_weights(a,3)
    except ValueError as error:
        if str(error) != 'inconsistent order conditions':
            raise
    else:
        raise AssertionError('inconsistent nonlinear system accepted')
    print('PASS: seven exact Python/Rust tableaux, weights, order equations and stability polynomials')


if __name__ == '__main__':
    main()
