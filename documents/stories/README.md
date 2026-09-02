# User stories — corpus / reading-companion track

Ordinary-language stories that drive this repo through PDD (`pdd intent`).
Each file is one story, written to the PDD drafting rule: **explicit
MUST / MUST NOT / Never, plus one concrete example**. The planner only captures
constraints phrased as obligations — prose without them yields zero invariants.

## Status

> **Research-stage correction — 2026-09-01:** stories 12–14 are retained as
> historical/provisional experiment sketches, but they are not approved for
> `pdd intent apply`. Their candidate model, dataset, objective, and compute plan
> remain unsettled. Use the comprehensive
> [physics-generative-surrogates handoff](../research/physics-generative-surrogates/README.md)
> and graduate the work back into stories only after the research criteria there
> are satisfied. The recorded intent IDs below are provenance, not authorization.

| # | Story | Intent ID | Invariants / examples captured |
|---|-------|-----------|-------------------------------|
| 1 | [Parse once, record what it produced](01-parse-once.md) | `as-a-researcher-building-propulsion-simulations--a5943e77` | 5 / 3 |
| 2 | [Configured corpus root, code-only repo](02-corpus-root.md) | `as-a-researcher-running-propulsion-tooling-when--a0b575b5` | 5 / 3 |
| 3 | [One queryable corpus index](03-corpus-index.md) | `as-a-researcher-with-a-dozen-parsed-textbooks-wh-68fe47cd` | 5 / 2 |
| 4 | [Reading companion, grounded answers](04-reading-companion.md) | `as-a-researcher-reading-toward-a-simulation-i-wa-ccdc8d0d` | 5 / 2 |
| 5 | [Corpus → cited simulation](05-corpus-to-simulation.md) | `as-a-researcher-when-the-corpus-gives-me-a-gover-6c9e01ed` | 5 / 2 |
| 6 | [SpaceX capability signal](06-spacex-capability-signal.md) | `as-an-engineer-deciding-which-physics-to-build-n-12a208d2` | 5 / 3 |
| 7 | [Corpus package transfer](07-corpus-package-transfer.md) | `as-a-researcher-with-one-gpu-machine-and-several-3fc67a86` | 5 / 2 |
| 8 | [Group-typed rigid body](08-group-typed-rigid-body.md) | `as-an-engineer-composing-simulations-across-doma-5b78cc0d` | 5 / 1 |
| 9 | [Adversarial physics-ML check](09-adversarial-physics-ml-check.md) | `as-an-engineer-betting-against-physics-specific--5d2e5d55` | 5 / 2 |
| 10 | [Modern LLM architecture](10-modern-llm-architecture.md) | `as-an-engineer-who-intends-to-fine-tune-rather-t-3794c1fd` | 5 / 4 |
| 11 | [RL foundations](11-reinforcement-learning-foundations.md) | `as-an-engineer-who-does-not-yet-know-reinforceme-5d5e58f3` | 5 / 3 |
| 12 | [Stiff-chemistry source-term surrogate](12-stiff-chemistry-source-term-surrogate.md) | `as-an-engineer-accelerating-reacting-flow-simula-aac8d0ba` | 5 / 2 |
| 13 | [Video-tokenizer transfer ablation](13-video-tokenizer-transfer-ablation.md) | `as-an-engineer-testing-whether-natural-video-pre-5de6038a` | 5 / 2 |
| 14 | [Full-field rollout surrogate](14-full-field-rollout-surrogate.md) | `as-an-engineer-evaluating-a-pretrained-generativ-e26a6b4f` | 5 / 2 |

Every story plans as `characterize_then_adopt` (conventional brownfield), and
every one asks the same question: *which current behavior should we lock down
with tests before changing anything?*

Dependencies: 2 → 1 → 3 → 4 → 5, with 7 following 1 and 8 standing alone.
Story 3 needs the completion record from story 1 ("never index a source that has
no recorded complete parse"); stories 4 and 5 need the locators from story 3;
story 7 ships story 1's parse record between machines. Stories 6 and 8 depend on
nothing here and can be applied in any order.

The physics-ML experiment order is story 12 first and story 13 → story 14
second, with story 9 adversarially checking both tracks. Stories 10 and 11 are
supporting studies; neither blocks the source-term or tokenizer experiments.

The vision these serve — scope, doctrine, sequencing, non-goals — is in
[../CHARTER.md](../CHARTER.md). That document holds what is not yet testable;
when a paragraph there becomes checkable it graduates into a story here.

## Re-planning a story

`plan` is read-only — no model call, no file changes, safe from any session:

```bash
pdd intent plan documents/stories/01-parse-once.md
pdd intent plan documents/stories/01-parse-once.md --json    # for a harness
```

The intent ID is a hash of the exact text. **Editing a story changes its ID**,
so re-plan and update the table above whenever a story changes.

## Applying a story

`pdd intent apply` makes model calls and must be run **in a terminal, by a
human** — the default model is `interactive_only`, so it hangs in a
non-interactive agent session. Apply also refuses an intent ID that differs from
the one reviewed, so run `plan` first and confirm the ID matches this table.

```bash
PDD_ALLOW_INTERACTIVE=1 pdd intent apply documents/stories/01-parse-once.md --characterized
```

Drop `--characterized` and apply will stop to ask for characterization tests
first. Write those tests against **current** behavior before passing the flag.

## Known gap for this repo

`pdd/data/language_format.csv` gives C++ empty `run_command` / `run_test_command`
and has no `.cu` row at all, so for the C++/CUDA parts of this repo `generate`
and `example` work but the automated crash/fix/verify loop has nothing to build
and run. The Python stories here (1–5) are unaffected.

## Characterization (stories 1 and 2)

`Corpus/unit_tests/test_corpus_layout_characterization.py` is the safety net for
stories 1 and 2. It is read-only, skips when `PROPULSION_CORPUS_ROOT` is unset or
the drive is unmounted, and asserts current behavior rather than desired
behavior — a failure means the corpus moved under the stories.

```bash
PROPULSION_CORPUS_ROOT=<CORPUS_ROOT> .venv/bin/python -m pytest Corpus -q
# 32 passed
```

Two findings from writing it went straight back into story 1:

- **Parsing has stages, not a boolean.** Five of the seven parsed books carry
  `reconciled/equations_resolved.json`; HorowitzHill-ArtOfElectronics3e and
  Arnold-MathematicalMethodsClassicalMechanics-2e stop after reconciliation.
- **A stage that produced nothing is indistinguishable from a failure.**
  Arnold holds the same `nougat_out/`, `reconciled/` and `.marker.md` as
  Lieuwen and reconciled 0 agreements, 0 conflicts and 1 marker-only equation.
  It is tempting to read that as a failed parse — an earlier note in this repo
  did — but the OCR is sound and the extracted text is full size. The book
  numbers no equations at all, so the tag-based reconciler has nothing to align
  on: 0 `\tag{}` in both engines' output against 843 for Goldstein. Neither
  directory presence nor a low equation count separates this from a real
  failure, which is why the record has to name the stage and its conclusion.

`test_no_source_carries_a_parse_completion_record_today` is the baseline gap, not
a requirement — delete it when story 1 lands.

## Decisions taken

- **Propulsion wraps, Monoclaw parses.** The GPU OCR pipeline stays at
  `Monoclaw/Python/ocr-compare`; Propulsion grows a thin corpus layer that owns
  the completion record, the corpus root and the index, and shells out when a
  parse is genuinely needed. Its location is configuration, not a fixed relative
  path.
- **Characterize 1–2, apply 3–5 direct.** Stories 3, 4 and 5 add new modules;
  there is no current behavior to preserve.

## Note on stories 6–8

These came out of the charter and are not yet characterized.

- **6 (SpaceX signal)** is the only story that reaches the network. Its MUST NOTs
  exist because a careers page is copyrighted, goes stale, and is a lagging
  indicator: the record keeps the extracted capability and the URL, never the
  source text, and every harvest is dated so the signal reads as a trend.
- **7 (package transfer)** is sized by measurement, and packages are expected to
  get large. The seven parsed books occupy 437 MB, but 315 MB of that is
  `reconciled/pages` and `reconciled/sheets` — PNG strips rendered for the
  vision-resolution pass and regenerable from the source — so a parsed-products
  package for all seven is about 122 MB. Original source documents are a
  separate, explicit choice at export time: they are not regenerable, the corpus
  holds 61 of them totalling 826 MB, and 13 are `.djvu` rather than `.pdf`
  (Sidi's among them), so selection is by content and never by extension. The
  package file lives outside the repository and is never committed, by LFS or
  otherwise; import surfaces it under the receiving machine's own corpus data
  directory, which is not the `Data/` fixture directory inside this repo.
- **8 (group-typed rigid body)** does not re-derive anything.
  `LaTeXandpdfs/SO3_SU2_Quaternions.tex` and `Cosmos/QuaternionConventionLab/`
  already fix the mathematics and the five conventions; this story makes that
  contract binding on `Cosmos/Source` and `CombustionInstability`, enforced by a
  double-cover property test rather than by a PDF.

## Note on stories 9–14 (the physics-ML bet)

These six serve [../PHYSICS-ML-BET.md](../PHYSICS-ML-BET.md), which holds the
thesis, the evidence, and the falsifiers. Read that first — several of the MUST
NOTs here only make sense against it.

Their requirements are unusually specific because they are derived from failures
found by reading the two repositories cited as evidence for the bet:

- **A policy whose actions never reach the simulator.** In one repo the action is
  written to a field nothing reads, while physics steps on random torques. Story
  11 requires an assertion for exactly this.
- **An update signal that is identically zero.** The same repo averages z-scored
  returns, which sums to zero by construction, giving per-weight updates near
  1e-19. Story 11 requires reporting update magnitude next to the reward curve.
- **Discarding the thing under test.** The other repo replaces a pretrained
  model's token embeddings and LM head with identity, then reports on transfer.
  Story 13 generalizes the lesson correctly for physics fields: channel adapters
  may change, but every retained, inflated, replaced, frozen, and trained
  parameter must be recorded, and the scratch control must be architecturally
  identical.
- **A missing baseline.** That same repo's own numbers show a 700 K from-scratch
  model beating the 410 M pretrained one by about 5x. Story 13 requires both
  initialization baselines on the same axes, and story 14 adds specialist
  field-surrogate baselines.

Story 12 is the engineering-first track: it keeps the classical solver and
learns only the stiff chemistry substep. Stories 13 and 14 are the research
track: the small tokenizer-transfer ablation gates the expensive full-field
autoregressive experiment. The RTX 3060 is useful for story 13 after the
model-independent harness exists; no 4 B checkpoint or cloud allocation belongs
in the first milestone.

Story 10 is deliberately narrow: the attention ladder already exists in CuLLM
from scalar through WMMA to CuTe, so the study covers only the modern stack
(mixture-of-experts routing, grouped-query attention, RoPE, normalisation
placement) and explicitly forbids rebuilding attention.
