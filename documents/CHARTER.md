# Charter — a multi-physics library, application, and harness

This document holds the parts of the vision that are **not** testable outcomes:
scope, doctrine, sequencing, non-goals. Testable outcomes live in
`documents/stories/` and go through PDD. Nothing here is a requirement; when a
paragraph here becomes checkable, it graduates into a story and is deleted from
this file.

## The three artifacts

**Library** — the physics, as reusable modules with declared mathematical
structure. This repository already holds nine domains (`Cosmos` for
astrodynamics and GNC, `CUDACFD`, `CombustionInstability`, `cantera_stuff` for
chemical kinetics, `ccdroplet`, `EM`, `Physique`, `Stunticons`, `T1000`). The
library is not greenfield; it is those domains given a common spine.

**Application** — a runnable multi-physics simulation composed from library
modules, eventually a full vehicle and eventually a full launch enterprise.

**Harness** — what lets an agent run a simulation, read its output, and check a
claim against a source. The harness is why the corpus work (stories 1–5) comes
first: an agent that cannot cite cannot be checked.

## What "AI-first" means here, operationally

Not "has an LLM in it." Four concrete properties, each already a story or a
direct consequence of one:

1. **The corpus is the knowledge substrate.** Parsed textbooks are a first-class
   input, parsed once and addressable by locator (stories 1–3).
2. **Every claim is citable.** Modules derived from the literature carry
   resolvable citations, and the check fails when a citation goes stale
   (story 5).
3. **Prompts are the source of truth.** Code is regenerated from stories through
   PDD, not hand-patched (`documents/stories/README.md`).
4. **The agent is held to the same standard as the engineer.** It answers from
   the corpus or says the corpus does not cover it (story 4).

## Doctrine: name the structure before writing the numerics

The stance is an engineering-physics engineer who has had lifetimes to become
fluent in pure mathematics, and who refuses to spend that fluency on decoration.
It cashes out as three habits:

- **State the group, space, or category first.** Before a rigid-body integrator
  is written, it is settled that attitude lives in `SO(3)`, that unit quaternions
  are `SU(2)` double-covering it, and that a 6-DOF state lives in `SE(3)` —
  with the covering map written out, not gestured at. `LaTeXandpdfs/SO3_SU2_Quaternions.tex`
  is that treatment for rotations. Story 8 makes it binding on the code.
- **Name every convention at an API boundary.** "JPL convention" is not a
  specification. `Cosmos/QuaternionConventionLab/README.md` names five separate
  choices that "JPL" leaves open. Every such boundary in the library gets the
  same treatment.
- **Cite the source, in the code.** Rigor that lives only in a PDF is not
  enforced. Story 5 turns citations into a check.

The rigor is instrumental. It earns its place where an unnamed convention or an
unstated chart causes a real defect — sign errors in attitude propagation,
frame inversions, a Lie-algebra step taken in the wrong chart. It is not an
invitation to categorify code that works.

## Sequencing: spine, then domains

The scope named in the request — combustion CFD, thermal, GNC, propulsion,
fluid mechanics, chemical kinetics, chemistry, plus manufacturing — is larger
than any one person ships as a unit. It is tractable in this order:

- **Track A — corpus and harness** (stories 1–5, 7). The knowledge substrate and
  the ability to move it between machines. *In progress.*
- **Track B — mathematical spine** (story 8). Group-typed state, one quaternion
  convention, conversions property-tested against the double cover. Touches
  every domain, so it comes before the domains multiply.
- **Track C — domain consolidation.** Bring the nine existing directories onto
  the spine one at a time, each with characterization tests first. No new domain
  starts until an existing one has landed.
- **Track D — composition.** Multi-physics coupling: combustion to acoustics to
  structure, propulsion to GNC. Requires B and C.
- **Track E — enterprise modelling.** Factories, manufacturing flow, launch
  cadence. Deterministic where reality is deterministic; stochastic where it is
  not. **Open question, not yet a requirement:** which quantities are genuinely
  random versus merely unmeasured. Track E does not start until D exists,
  because a factory model with no vehicle model to feed has nothing to be
  validated against.

## Non-goals

- **Not a general-purpose physics engine.** Depth on the vehicle and its
  physics, not breadth across all simulation.
- **Not a replacement for the corpus.** The library implements what the books
  say; it does not restate them. Derivations live in `LaTeXandpdfs/` and in the
  corpus, and are cited.
- **Not offline-hostile.** The corpus, the library, and the harness work with no
  network. Only the SpaceX signal harvest (story 6) reaches out, and only when
  asked.
- **Not a rewrite.** Existing domain code is consolidated behind characterization
  tests, never replaced wholesale.

## On "useful to SpaceX"

Story 6 turns the newsroom and careers pages into a dated, cited capability
signal. Two honest limits on that signal, recorded here so the story is not
over-read: a job posting is a **lagging** indicator, describing work already
scoped and staffed; and a capability list is not a design. The signal is useful
for deciding which of tracks C and D to sequence next, and for naming gaps. It
is not a substitute for the physics being correct, which is what tracks A and B
buy.
