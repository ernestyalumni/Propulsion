# Hairer reference harness (DOPRI5 / DOP853 golden vectors)

Runs E. Hairer's own Fortran codes `dopri5.f` (version of April 25, 1996)
and `dop853.f` (version of October 11, 2009) on fixed test problems. It writes
every accepted step, dense-output samples and the final statistics to
`golden/hairer_reference/<method>_<case>.tsv`. The Rust embedded Runge–Kutta
tests read these files. As with the other golden vectors in `golden/`, the
Fortran build is the emitter and Rust is the reader: never edit the TSVs by
hand, and never regenerate them from Rust.

## Licence: fetched, not vendored

Hairer's codes are "free subject to the licence agreement" on
<http://www.unige.ch/~hairer/software.html>. They are **not committed** to this
repository. `fetch_hairer_codes.sh` downloads them into
`Cosmos/Rust/target/hairer_reference/` (gitignored) and checks these digests:

| file | sha256 |
|---|---|
| `dopri5.f` | `e4fbbe080508134f743d9934e76e45f49f03a320b7b024273757a7c278804c95` |
| `dop853.f` | `8bfc0a954a38af1a458f5ab8025fb09c923373a936f52ff61e32f1a15c8f298a` |

If a digest does not match, the script deletes the download and exits with an
error. Hairer may have published a new version; read the diff before you
update the digest.

## Regenerate

```bash
cd Cosmos/Rust
tools/hairer_reference/emit_golden.sh                    # downloads the codes
tools/hairer_reference/emit_golden.sh --from DIR         # or copy local dopri5.f/dop853.f (still hash-checked)
```

Compiler flags: `gfortran -O2 -ffp-contract=off -fno-fast-math -cpp -std=legacy` for
`dopri5.f`/`dop853.f`; the driver (which holds the right-hand sides) is compiled
at `-O0 -ffp-contract=off -fno-fast-math -cpp`. At `-O1` and above gfortran fuses
`cos(x)` and `sin(x)` into glibc `sincos`, which differs from separate `cos`/`sin`
(and from Rust's `f64::cos`/`f64::sin`) in the last bit for about 0.07% of
arguments; `emit_golden.sh` refuses to write golden files if `sincos` is linked.
With that, the Rust integrators reproduce every file bit for bit on x86-64 Linux.
There is no `-march=native` and no FMA contraction, so the arithmetic is plain
IEEE double. Each method gets its own executable, because both Fortran files
define `HINIT`. The provenance header of each TSV records the source digests,
the driver digest, the compiler version and the flags.

## Settings for every run

- `WORK(1) = epsilon(1d0)` = 2^-52 (unit roundoff).
- `IWORK(3) = -1`: silent. In this mode, when stiffness is detected the code
  *aborts* with `IDID = -4`.
- `IOUT = 2` (dense output) and `IWORK(5) = N` (dense output for all components).
- `ITOL = 0` (scalar `rtol = atol = tol`) unless a case says otherwise.
- `H = 0`, so the starting step comes from `HINIT`, unless a case says otherwise.
- All other `WORK`/`IWORK` entries are zero, which selects the defaults:
  - DOPRI5: safe 0.9, fac1 0.2, fac2 10, beta 0.04.
  - DOP853: safe 0.9, fac1 0.333, fac2 6, beta 0.
  - Both: nmax 100000, nstiff 1000, hmax = |xend − x|.

## Problems

**Arenstorf orbit** (HNW I, eq. (0.1), pp. 129–130).
- State: `y = (y1, y2, y1', y2')`, with μ = `0.012277471d0`.
- Initial value: `y0 = (0.994, 0, 0, -2.00158510637908252240537862224)`.
- Interval: from 0 to `xend = 17.0652165601579625588917206249`.

The right-hand side is evaluated in exactly this floating-point order. A Rust
twin has to use the same order to match to the last bits.

```
mu_prime = 1.0 - mu
r1 = (y1 + mu)*(y1 + mu) + y2*y2;              r1 = r1*sqrt(r1)
r2 = (y1 - mu_prime)*(y1 - mu_prime) + y2*y2;  r2 = r2*sqrt(r2)
f1 = y3
f2 = y4
f3 = ((y1 + 2.0*y4) - (mu_prime*(y1 + mu))/r1) - (mu*(y1 - mu_prime))/r2
f4 = ((y2 - 2.0*y3) - (mu_prime*y2)/r1) - (mu*y2)/r2
```

`D^{3/2}` is computed as `r*sqrt(r)`, following Hairer's own drivers, not as
`pow(r, 1.5)`. `sqrt` is correctly rounded everywhere; `pow` is not.

**Prothero–Robinson** (a stiff scalar problem): `f = (-1000*(y - cos x)) - sin x`,
with `y(0) = 1` on `[0, 10]`. The result depends on glibc `cos`/`sin`, which
Rust's `f64::cos`/`sin` also call on Linux.

## Cases (each run with both `dopri5` and `dop853`)

| case | settings |
|---|---|
| `arenstorf_tol1e-4`, `_tol1e-7`, `_tol1e-10` | rtol = atol = tol |
| `arenstorf_backward_tol1e-7` | from x = xend back to 0, starting at y0 |
| `arenstorf_componentwise` | ITOL=1, atol = (1e-8, 1e-8, 1e-6, 1e-6), rtol = (1e-7, 1e-7, 1e-5, 1e-5) |
| `arenstorf_given_step` | tol 1e-7, `WORK(7) = 1e-3` (initial h), `WORK(6) = 0.5` (hmax) |
| `prothero_robinson_stiff` | tol 1e-6 |

## TSV format

Fields are tab-separated. Numbers are printed as `ES25.16E3`, which is 17
significant digits, so every double round-trips exactly.

```
# provenance and case settings (method, case, n, x0, x_end, y0, itol, rtol, atol, work(1)/(6)/(7))
step	<nr>	<xold>	<x>	<y_1> … <y_n>          one per SOLOUT call; nr = 1 is the initial point (xold = x)
dense	<nr>	<theta>	<x_theta>	<y_1> … <y_n>   for nr ≥ 2 and theta ∈ {0.25, 0.5, 0.75}
stats	<idid>	<nfcn>	<nstep>	<naccpt>	<nrejct>	<x_final>	<h_final>
```

How the dense samples are computed:
- `x_theta = xold + theta*(x - xold)`.
- The value is `CONTD5`/`CONTD8(i, x_theta, …)`.
- Inside those functions, Hairer evaluates the polynomial at
  `s = (x_theta - xold)/h`, where `h` is the step size actually used (`HOUT`).
  That `h` can differ from `x - xold` in the last bit. To match bitwise,
  recompute `s` from `x_theta` the same way, not from `theta`.

`h_final` is `WORK(7)` on return, which is the next proposed step size.

## Hairer's counting quirks (matter when comparing statistics)

- `nfcn` counts 2 for the initial `f(x0, y0)` plus the `HINIT` trial, even when
  an initial step is given. Without a stiffness abort:
  - DOPRI5: `nfcn = 2 + 6*nstep`.
  - DOP853 with dense output: `nfcn = 2 + 11*nstep + naccpt + 3*naccpt`.
  - DOP853 aborting with `IDID = -4`: the last accepted step makes its FSAL call
    but skips the 3 dense-output stages, so `nfcn = 2 + 11*nstep + naccpt + 3*(naccpt - 1)`.
- `nrejct` excludes rejections that happen before the first accepted step.
  `nstep - naccpt - nrejct` is therefore the number of initial rejections.
