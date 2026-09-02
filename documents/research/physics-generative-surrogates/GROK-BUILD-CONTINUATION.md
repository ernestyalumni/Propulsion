# Prompt for the next Grok Build session

Copy the prompt below into a new Grok Build session that can read the same
workspace.

---

You are continuing a pre-story research investigation in Ernest Yeung’s
Propulsion repository. Do not begin implementation, download model weights, rent
cloud GPUs, or run PDD apply. The work is deliberately not yet at user-story
level.

Repository:

`/media/propdev/Expansion/openclaw/.openclaw/workspace/repos/Propulsion`

Start by reading these files completely, in order:

1. `documents/research/physics-generative-surrogates/README.md`
2. `documents/research/physics-generative-surrogates/SESSION-HANDOFF.md`
3. `documents/research/physics-generative-surrogates/EVIDENCE-AND-OPEN-QUESTIONS.md`
4. `documents/PHYSICS-ML-BET.md`
5. `LaTeXandpdfs/SourceTermSurrogate.tex`
6. `.claude/agents/surrogate-advocate.md`
7. `.claude/agents/transfer-advocate.md`

The existing files `documents/stories/12-*`, `13-*`, and `14-*` are provisional
experiment sketches. Do not treat them as approved requirements and do not
apply them through PDD. Use them only to recover useful invariants.

## Objective

Narrow the question “Can an LLM or fine-tuned open model become a numerical
generator for physics?” into the smallest defensible experimental decision.

Keep these targets separate:

- text-LLM authoring of physics code and objectives;
- supervised stiff-chemistry flow-map/source-term surrogate;
- full-field spatiotemporal numerical surrogate;
- visually plausible RGB world generation.

Only the middle two are numerical surrogates.

The working thesis is that a standard pretrained generative architecture may be
preferable to inventing an exotic physics architecture, and that pretrained
weights may transfer when the modality shares structure with the target. This
is a hypothesis, not a conclusion. Attack it honestly.

## Research tasks

1. Recheck the current open-weights landscape as of the session date using
   primary sources: official model cards, repositories, papers, and licenses.
2. Audit `nvidia/Cosmos3-Edge` component by component:
   - exact parameter counts by component;
   - VAE channel and latent shapes;
   - generator conditioning interface;
   - which path handles video and action generation;
   - official full-fine-tune and LoRA support;
   - minimum realistic inference and adapter-training memory;
   - what can be isolated on an RTX 3060 12 GB.
3. Compare Edge with at least:
   - the Cosmos-1.0/PhysiX stack as a historical transfer baseline;
   - the identical selected architecture trained from scratch;
   - a strong current specialist surrogate such as FNO/TFNO or a more suitable
     operator for the selected field problem;
   - a small convolutional and persistence baseline;
   - the classical solver or practical reduced-order incumbent.
4. Determine whether Cosmos3’s diffusion generator is fundamentally mismatched
   to low-latency deterministic time stepping, or whether its probabilistic
   output supplies a valuable uncertainty/ensemble use case.
5. Inspect local candidate data sources before recommending The Well by default:
   - `CombustionInstability/`
   - `CUDACFD/`
   - `Data/TurbulentCFDExampleCases/`
   - `Cosmos/Source/Numerical/`
6. Propose no more than three sharply different candidate experiments. For each,
   state:
   - user or engineering value;
   - input/output state and units;
   - data source and generation cost;
   - transferred weights and modified layers;
   - scratch and specialist baselines;
   - physical and performance metrics;
   - local and cloud compute estimate;
   - preregistered go/no-go criterion;
   - the strongest reason the experiment may be pointless.
7. Force-rank the candidates. Recommend exactly one next research action that
   can be done without downloading the full model.

## Evidence requirements

- Cite a primary, openable source beside every externally checkable claim.
- Record paper and checkpoint versions or revisions.
- Identify the denominator behind every speedup or “x-times better” number.
- Clearly label inference from source material as inference.
- Do not infer numerical accuracy from visual quality.
- Do not claim solver acceleration without wall time at matched acceptable
  physical error.
- Do not claim transfer if interface changes discarded the relevant pretrained
  weights.
- Do not describe return-conditioned behavior cloning as RL.
- Do not claim LoRA supplies conservation, units, a new channel interface, or
  numerical stability by itself.
- Preserve the possibility that a small from-scratch model is the correct
  answer.

## Required output

Write a new research memo under:

`documents/research/physics-generative-surrogates/`

Suggested name:

`CANDIDATE-EXPERIMENTS-YYYY-MM-DD.md`

Do not modify the provisional user stories. End the memo with one of three
explicit states:

1. `NOT READY` — a key decision or evidence source is still missing;
2. `READY FOR BENCHMARK CONTRACT` — candidate, data, baselines, and metrics are
   fixed, but still no product story;
3. `READY FOR USER-STORY REVIEW` — an independently testable behavior and
   acceptance oracle now exist.

Most likely, the correct state after one research pass is 1 or 2. Do not force
graduation.

---
