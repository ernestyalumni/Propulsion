# Study handoff — Hairer I, Exercise II.1.1

Recorded 2026-09-15 at Ernest's request. He is pausing to work on other
things and wants to continue working out the solution himself, with an
assistant alongside him. This is a progress record, not another solutions
manuscript.

## Read these first

1. [STYLE.md](STYLE.md): the evolving exposition conventions.
2. [topics/runge-kutta.tex](topics/runge-kutta.tex): the authoritative
   manuscript. Find the exercise heading
   `\begin{exercise}[Hairer I, II.1.1, p.~141 \cite{HNW1993}]`,
   around line 1152; Ernest's solution starts around line 1165.
3. The earlier AI-written worked solution, as a reference for the remaining
   reasoning, not a replacement for Ernest's write-up:
   `/home/propdev/.openclaw/workspace/hairer-work/rk-tutoring/hairer-i-ii-1-exercise-1-solution.md`.
   Its exercise statement was paraphrased before Ernest specified the
   verbatim convention; use the manuscript's quoted exercise and hint.

Repository root on this machine:
`/home/propdev/.openclaw/workspace/workspace2/repos/Propulsion`.
The canonical solution belongs inline in the shared topic, which is included
by both master.tex and the Numerical Recipes companion.

## Where Ernest stopped

He has written the following progression:

- Recall the vector stage formulas from Eq. (66), whose stable label is
  `Eq:ERKmethodsSthStageMyNotation`.
- Specialize to \(\mathbf y'=\lambda\mathbf y\), with scalar constant
  \(\lambda\), retaining the \(\mathbf k_i,\mathbf y_n\) notation.
- Work through the substitutions for \(\mathbf k_1,\mathbf k_2,\mathbf k_3\),
  and introduce \(z=h\lambda\).
- Motivate and define
  \[
  P_1(z)=1,\qquad
  P_i(z)=1+z\sum_{j=1}^{i-1}a_{ij}P_j(z).
  \]
- Explicitly check the first two cases, then write the induction
  substitution leading to the intended general representation
  \[
  \mathbf k_i=\lambda\mathbf y_nP_i(z).
  \]
  The last displayed line in the present draft gives this at stage \(s\).

This is the stopping point. The draft still has local transcription details
to tidy; it is not a claim that the full exercise is complete or that the
argument has received a final mathematical review.

**The next mathematical task is the degree bound**
\(\deg P_i\leq i-1\). Ernest has not yet written it in this solution.
He has also not yet formed the output polynomial \(R\), applied order \(s\),
or completed the derivative comparison.

## Resume in this order

1. Re-read the final base-case and induction lines with Ernest and tidy the
   few local slips listed below, preserving his sequence and wording.
   State explicitly that the induction hypothesis holds for every earlier
   stage \(j<i\).
2. Prove the degree bound by induction from the definition of \(P_i\).
   In the reference Markdown, resume at the paragraph beginning
   “Now suppose each earlier \(P_j\) has degree at most \(j-1\)” (around
   line 89).
3. Recall the final update from Eq. (66), including its source reference:
   \[
   \mathbf y_{n+1}=\mathbf y_n+h\sum_{i=1}^{s}b_i\mathbf k_i.
   \]
   Substitute the stage representation explicitly, factor out
   \(\mathbf y_n\), and identify \(R(z)=1+z\sum_i b_iP_i(z)\).
   Establish degree at most \(s\). Reference guide:
   “Substitute the stages into the final update” (around line 101).
4. Invoke the local-order definition, not an unstated accuracy fact.
   Stable labels are `Eq:RungeKuttaMethodOrderpConditionDefinition` and
   `Eq:RKLocalErrorOurNotation` (Eqs. (67)--(68) in the recent build).
   Start from exact data; hold the starting point, starting state,
   \(\lambda\), and tableau fixed while varying \(h\).
5. Work out the hint's derivatives with respect to \(h\), evaluated at zero,
   and compare with the exact exponential solution to determine the
   coefficients of \(R\). Reference guide:
   “Use order s and differentiate with respect to h” (around line 158).
   Distinguish degree at most \(s\) before comparison from degree exactly
   \(s\) after finding the nonzero highest coefficient.

Hairer's exercise is scalar; Ernest is using the vector notation from
Eq. (66), with \(\lambda\) still a scalar. Keep that distinction explicit
when reaching the comparison: specialize to the scalar problem or compare
scalar multipliers/components; never divide by a vector. Handle zero
initial data and zero \(\lambda\) before any cancellation requiring them
to be nonzero. For the final statement, map \(n,n+1\) back to Hairer's \(0,1\).

For an additional reference already in the notes, see
`Eq:RKStabilityPolynomial` and `Eq:RKStabilityPolynomialMatrix`,
recently Eqs. (114)--(115). Cite labels rather than assuming the printed
equation/page numbers cannot change.

## What to retain from Ernest's writing

The manuscript is the current style example; the full AI Markdown is a
reasoning aid. The comparison gives concrete guidance:

- Start by naming and displaying the equation being used. Ernest begins
  with “Recall Eqn.” and the actual vector stage formulas before specializing.
- Keep the substitution trail visible: original stage evaluation, problem
  substitution, earlier-stage substitution, factoring, and collected powers.
  The intermediate equalities are part of the explanation.
- Let explicit cases motivate notation. His \(\mathbf k_2,\mathbf k_3\)
  calculations precede “This suggests ...” and the definition of \(P_i\).
- After a definition, verify its first instances by expanding it. His
  “Notice then that” checks both \(P_1\) and \(P_2\) against the computed stages.
- Pair “by induction” with the actual substitution and factoring, and make
  the hypothesis/range explicit when refining the draft.
- Use short connecting prose with detailed mathematics. Source each starting
  fact at its use, including the update formula and order definition still
  to come. Break long displays at meaningful equalities for readability
  without deleting the intermediate steps.
- Preserve his vector notation and order of reasoning unless a necessary
  change is explained. These preferences remain collaborative and evolving.

Do not silently paste the remaining worked answer into the manuscript on
resumption. Ernest wants to work through it himself; follow his next request
and help at the step where he is working. Do not mark the exercise, reading
checkpoint, or adaptive-step preparation complete from the existence of an
AI-written solution.

## Local cleanup to revisit, not changed during this handoff

Line numbers refer to the source inspected on 2026-09-15:

- Around line 1195, `a{32}` in the substituted \(\mathbf k_3\) line should
  be the stage coefficient `a_{32}`.
- Around line 1210, the \(P_2\) check has
  `\lambda \mathbf (1 + za_{21}P_1(z))`; the intended intermediate factor is
  `\lambda \mathbf y_n (1 + za_{21}P_1(z))`.
- Around line 1216, an intermediate `\mathbf y` needs its step subscript
  \(n\). Retain the argument \((z)\) on \(P_j\) and \(P_s\) when spelling out
  the substitution.
- Around lines 1177--1181, identify the derivative evaluation at \(x_n\)
  when equating it to \(\mathbf f(x_n,\mathbf y_n)\), with exact starting
  data if interpreting it as the exact solution's derivative.

These are cleanup items, not stylistic patterns to reproduce. The source
was left untouched for the pause.

## Artifact and repository state

The checkpoint is based on the current .tex source. Its hash differs from
the input recorded by the last PDF build, dated
2026-09-16T05:10:19Z (2026-09-15 in Los Angeles), so that PDF must not be used
as evidence that it includes this latest draft. No rebuild was requested
for this handoff.

When manuscript editing resumes, use the documented builder from the repo
root to refresh the live ReadingRoom artifacts:

~~~sh
python3 -B ReadingRoom/build_notes.py --output /home/propdev/.openclaw/workspace/Data/ReadingRoom/propulsion/artifacts
~~~

The working tree contains other manuscript changes and a staged ODE
migration/legacy-copy cleanup. Preserve them. This handoff neither stages
nor commits anything.
