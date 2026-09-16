# Runge–Kutta manuscript incorporation — 2026-09-12

Ernest requested incorporation, almost verbatim, of the RK discussion in
`LaTeXandpdfs/Propulsion.tex` from the section beginning at original line 5228
through the short order-conditions subsection, ending before adaptive
stepsize control at original line 5657. He also requested completion of the
TODO, comparison of both extra copies, clearer connecting exposition, and a
durable record of his notation and algebra style.

The shared chapter is `topics/runge-kutta.tex`. Both `master.tex` and
`companions/numerical-recipes.tex` include that same source. Its sequence
retains the IVP, appendix/Hairer notation comparison, stage and order
definitions, Euler/Heun/midpoint examples, RK4 derivation and tableau,
linear-system specialization, and opening order-conditions discussion.
The substantive additions complete the derivations and make the notation
changes explicit. This is an edited incorporation, not a claim of byte-for-byte
reproduction. Adaptive control and later manuscript sections are outside this
incorporation.

At the initial RK incorporation, the historical manuscripts remained unchanged.
The later authorized cleanup is recorded below. The existing staged ODE migration
and unrelated note/ReadingRoom changes were preserved. No commit or push is
part of this work. PDD ownership was inspected: the existing prompt specifies
the document/study tooling and three starter studies; this mathematical
chapter is an authored manuscript input. No code regeneration or completed
PDD synchronization is claimed.

## Extra-copy comparison

Both copies first appear in `8320fa16a376e2e7793af6c9e51f3d06520ca55a`,
dated 2023-10-04, authored by Ernest, with message “Try to untangle Propulsion
LaTeX.” That commit also substantially reduced the main manuscript.
`PropulsionExtraCopy2.tex` begins with a stray `=======` line. These facts are
consistent with saved material from untangling a duplicated/conflicted file;
the exact reason for keeping each copy is not otherwise documented.

| File | Lines | Bytes | SHA-256 before incorporation |
|---|---:|---:|---|
| `Propulsion.tex` | 10426 | 495377 | `7fbf5a8128689e37b82a455ea7c470ac319a30635e2d1d7a0aed5b8051609199` |
| `PropulsionExtraCopy.tex` | 8607 | 411194 | `b17cc6a9141fda9db261db678b27cdb5b234306d84d4bcd6f87dcab8e4db47c1` |
| `PropulsionExtraCopy2.tex` | 8586 | 410368 | `3e20853d7c761e7df12fec3fc34b95b910e1ae2b18c0a0c8b96357a4512ed757` |

Full-file sequence comparisons each found 25 nonidentical blocks. Their
copy-only material was inspected, not merely their file lengths:

- **Useful unique material:** both copies contain the same short
  “Exponentiation by Squaring” section absent from the main file (copy 1
  lines 5159–5171; copy 2 lines 5160–5172). It gives, for positive integer n,
  `x^n = x (x^2)^((n-1)/2)` for odd n and `x^n = (x^2)^(n/2)` for even n,
  with a Wikipedia pointer. A recursive implementation would also state the
  base case `x^0 = 1`. This is worth preserving, but belongs to a different
  numerical topic and was not inserted into the RK chapter.
- **No additional RK derivation:** the copies omit the main manuscript's
  Hairer tableau, definitions/notation translation, extra examples, and its
  expanded material after the original RK4 tableau. In this topic the main
  file is the developed source.
- Other copy-side differences are blank lines, the `examinng` typo where the
  main file has `examining`, an unnumbered display where the main file has
  `Eq:IntegrateConservedVector`, and the stray initial marker in copy 2.
  Most changes are material present only in the main manuscript, including
  later mathematical sections and bibliography additions.

Neither copy was deleted during the initial RK incorporation. Full unified
comparisons and the selected original
passage are retained in the workspace audit directory
`hairer-work/rk-notes-integration/`.

## Corrections and additions

1. Set the general IVP's initial point to `a` on `[a,b]`; retain `0` as its
   special case. Use a map `f: [a,b] × U → R^d`. Distinguish state dimension,
   mesh count and stage count.
2. Preserve the appendix's `alpha/beta/c` and Hairer's `c/a/b` as explicitly
   separate notations. Add a dictionary to our vector notation and NR §17.1's
   step-scaled stages. The collision in `c_i` is explained, not silently renamed.
3. Separate exact values from numerical values. Stages initialized at index n
   update `y_(n+1)`, including the Euler and RK4 examples. Correct the local
   error to compare the exact endpoint with the numerical endpoint, from an
   exact starting value.
4. Correct the count of lower-triangular stage couplings to `m(m-1)/2` and
   the iteration range to `0,...,M-1`. Require a positive stage count.
5. Explain the role of `c = A 1` and its stage-consistency interpretation.
   These row sums are an imposed simplifying convention, not an unrestricted
   defining requirement for every ERK tableau.
6. Complete the two-stage coefficient matching with component/Jacobian
   notation. Identify the second example as **explicit midpoint**: “RK2”
   alone does not distinguish it from other second-order methods.
7. Replace the incomplete third/fourth total derivatives with all product-
   and chain-rule terms, including mixed derivatives. Define the actual total
   derivative `D = partial_t + f · grad_x`.
8. Complete the RK4 TODO via time augmentation, explicit stage-displacement
   expansion, and term-by-term Taylor comparison. Derive all eight conditions
   through order four and translate them back to the original notation.
   Add the missing fourth row sum to the original eleven-equation list.
9. Solve the original choices `alpha_2 = 1/2`, `beta_31 = 0` explicitly for
   the classical RK4 tableau. Avoid treating a count of nonlinear equations
   as proof of independence or uniqueness.
10. State `y' = J y` with constant J before the linear-stage/stability-polynomial
    rewrite. Show the repeated substitutions and finite matrix inverse.
    Explain why the scalar stability polynomial alone does not establish
    nonlinear order.
11. Expand the final order-conditions stub: distinguish stage states from
    stage derivatives; retain time arguments unless the field is autonomous
    or augmented. Supply the compact matrix forms and a fixed-interval
    local-to-global error argument with explicit hypotheses.
12. Preserve meaningful existing equation labels and add labels at new
    substitutions. Add real bibliography entries for all new citation keys.
    Retain `HaWa2010` as a legacy key, while identifying the second edition's
    1996 publication and the later 2010 printing correctly.

## Sources and verification

The original PDFs were checked at NR printed pp.907–908 (PDF pp.931–932)
and Hairer I printed pp.134–135, 143 (PDF pp.147–148, 156). The cited
[Münster appendix](https://www.uni-muenster.de/imperia/md/content/physik_tp/lectures/ss2017/numerische_Methoden_fuer_komplexe_Systeme_II/rkm-1.pdf)
was also inspected; the incomplete row-sum list occurs there too. Book edition
records were checked against [Hairer's book page](https://www.unige.ch/~hairer/books.html).

`checks/runge_kutta.sage` checks all eight classical-tableau order conditions,
the stability polynomial, the component third/fourth derivatives against
independent total differentiation, and the full symbolic RK4 step against a
Taylor expansion on a specified nonlinear nonautonomous polynomial field.
It also checks fixed-step convergence for `y'=y-t^2+1`, `y(0)=1/2`.
These calculations supplement the general handwritten argument. They do not
certify every legacy solver, physical model, or equation in the master.

The installed Sage image is
`sha256:aeef59a8c17212357aeb83bd7f1cfac43bb1c789ae3fa2d887227c763bdfcac3`.
It runs with no network and read-only source mounts. The actual result and
source hashes are published beside the built PDFs as
`runge-kutta-verification.json`. All eight companion/master PDFs are rebuilt
through the existing builder into the configured ReadingRoom artifact
directory. Personal reading progress is independent and is preserved.

The exposition conventions are in [STYLE.md](../STYLE.md).


## Authorized extra-copy cleanup — 2026-09-12

Ernest subsequently authorized preserving the unique addition in the main
manuscript and running `git rm` on both extra copies if no useful information
would be lost. Rechecked the complete files before deletion.

The identical 13-line “Exponentiation by Squaring” section was inserted
verbatim into `LaTeXandpdfs/Propulsion.tex`, after the Numerical Computation
part heading and before Interpolation and Extrapolation. The useful passage
and its original reference are preserved; its mathematical content was not
rewritten as part of this cleanup.

A full ordered-line coverage check establishes that the updated main file
contains all 6,953 normalized nonblank lines of the first copy and all 6,938
of the second. Normalization only ignores blank lines, the stray `=======`
marker, added equation labels, numbered versus unnumbered display delimiters,
and the `examinng`/`examining` typo. Thus the deletion does not discard another
derivation or reference. The source snapshots and check record are retained in
`hairer-work/rk-notes-integration/extra-copy-cleanup/`; Git history also retains
the original copies.

The main-file addition and both `git rm` deletions are staged together.
Unrelated staged changes were preserved. Nothing was committed or pushed.


## Explicit derivative notation — 2026-09-12

Ernest requested full partial-derivative fractions by default throughout
documents/notes, reserving derivative shorthand for algebra extending over
several pages. Any shorthand must be defined with explicit fractions before
use, with local reminders after gaps. STYLE.md now records this convention.

The RK two-stage and third/fourth chain-rule calculations use full fractions.
The autonomous derivative dictionary expands all three state-derivative maps
and states the augmented coordinate ranges. Component superscripts, powers,
free/summed indices, fixed variables and evaluation points are made explicit.
The saved tutor summary and shorter foundations, force-one-form and Sutton
passages follow the same convention. Existing equation labels are retained.

The third derivative, auxiliary U/B/C definitions, derivative of B, derivative
of the Jacobian-times-U expression, and fourth derivative are term-for-term
identical to the prior manuscript after normalizing derivative notation.
All eight existing Sage checks pass again, and the scoped report has current
source hashes. All eight PDFs rebuild with resolved references and no overflow;
master portrait pages 30 and 32 and screen page 16 were visually inspected.
The Git index is unchanged. Audit snapshots and diffs are in the workspace
hairer-work/rk-notation-revision directory. No commit or push was made.


## Concise component reminders and contractions — 2026-09-13

Ernest refined the preceding convention: component reminders should be one
short sentence, and Einstein summation is assumed without explanation.
Vector-valued derivative expressions now differentiate with respect to a
scalar coordinate and contract against a scalar component, leaving boldface
to identify the vector-valued output. STYLE.md records the precise notation.

Applied to the RK Taylor expansions and the tutor summary. Removed repeated
component/summation explanations in RK and foundations; all equation labels
are retained. Three Taylor equation blocks change only in their contraction
notation, and the checked component/order-condition blocks are unchanged.
The prior Sage execution and its original input hashes are retained with
a subsequent notation-review record; the symbolic check source is unchanged.
All eight PDFs rebuild with resolved references and no overflow; screen
master page 15 was visually inspected. The staged index is unchanged.
Audit snapshots, diffs, and comparison evidence: the workspace directory
hairer-work/rk-contractions-2026-09-13/. No commit or push was made.


## Explicit global-error proof — 2026-09-13

Ernest requested the mathematical work underlying the error recurrence and
geometric-series bound. Replaced the condensed concluding paragraph with
an explicit RK4 step-map definition retaining the starting independent
variable, two separate uniform hypotheses, and definitions of the comparison
states and perturbation-growth constant L. The proof now inserts the zero
term, regroups the errors, substitutes the two bounds, expands E1/E2/E3,
proves the finite-sum bound by induction, cancels the shifted geometric sums,
and derives the fixed-interval result with a separate L=0 case.

The earlier manuscript and all existing labels are retained. This proof
uses the two stated estimates as hypotheses; it does not claim fourth-order
accuracy from Lipschitz continuity alone. STYLE.md records the distinction
between brief notation reminders and fully worked algebra.

All eight PDFs build with resolved references and no overflow. Source/output
hashes are current; screen pages 19–20 and portrait page 39 were visually
inspected. The complete proof spans screen pages 19–21 / portrait pages
37–40. Existing scoped Sage execution results are retained with separate
exposition-revision provenance; the check source is unchanged. Git index
unchanged; no commit or push. Audit snapshots and diffs are in the workspace
hairer-work/rk-global-error-2026-09-13 directory.
