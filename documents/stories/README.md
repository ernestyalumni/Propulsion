# User stories — corpus / reading-companion track

Ordinary-language stories that drive this repo through PDD (`pdd intent`).
Each file is one story, written to the PDD drafting rule: **explicit
MUST / MUST NOT / Never, plus one concrete example**. The planner only captures
constraints phrased as obligations — prose without them yields zero invariants.

## Status

| # | Story | Intent ID | Invariants / examples captured |
|---|-------|-----------|-------------------------------|
| 1 | [Parse once, record what it produced](01-parse-once.md) | `as-a-researcher-building-propulsion-simulations--7b053a29` | 5 / 3 |
| 2 | [Configured corpus root, code-only repo](02-corpus-root.md) | `as-a-researcher-running-propulsion-tooling-when--a0b575b5` | 5 / 3 |
| 3 | [One queryable corpus index](03-corpus-index.md) | `as-a-researcher-with-a-dozen-parsed-textbooks-wh-68fe47cd` | 5 / 2 |
| 4 | [Reading companion, grounded answers](04-reading-companion.md) | `as-a-researcher-reading-toward-a-simulation-i-wa-ccdc8d0d` | 5 / 2 |
| 5 | [Corpus → cited simulation](05-corpus-to-simulation.md) | `as-a-researcher-when-the-corpus-gives-me-a-gover-6c9e01ed` | 5 / 2 |
| 6 | [SpaceX capability signal](06-spacex-capability-signal.md) | `as-an-engineer-deciding-which-physics-to-build-n-12a208d2` | 5 / 3 |
| 7 | [Corpus package transfer](07-corpus-package-transfer.md) | `as-a-researcher-with-one-gpu-machine-and-several-b9f85b9c` | 5 / 2 |
| 8 | [Group-typed rigid body](08-group-typed-rigid-body.md) | `as-an-engineer-composing-simulations-across-doma-5b78cc0d` | 5 / 1 |

Every story plans as `characterize_then_adopt` (conventional brownfield), and
every one asks the same question: *which current behavior should we lock down
with tests before changing anything?*

Dependencies: 2 → 1 → 3 → 4 → 5, with 7 following 1 and 8 standing alone.
Story 3 needs the completion record from story 1 ("never index a source that has
no recorded complete parse"); stories 4 and 5 need the locators from story 3;
story 7 ships story 1's parse record between machines. Stories 6 and 8 depend on
nothing here and can be applied in any order.

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
- **A failed parse is indistinguishable from a good one by directory presence.**
  Arnold holds the same `nougat_out/`, `reconciled/` and `.marker.md` as
  Lieuwen, but reconciled 0 agreements, 0 conflicts, 1 marker-only equation, and
  one nougat page repeated 24 times. A presence heuristic would mark that failed
  parse complete and permanently block the redo.

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
- **7 (package transfer)** is sized by a real measurement. The seven parsed books
  occupy 437 MB, but 315 MB of that is `reconciled/pages` and `reconciled/sheets`
  — PNG strips rendered for the vision-resolution pass and regenerable from the
  PDF. Excluding them takes a full-corpus package to about 122 MB.
- **8 (group-typed rigid body)** does not re-derive anything.
  `LaTeXandpdfs/SO3_SU2_Quaternions.tex` and `Cosmos/QuaternionConventionLab/`
  already fix the mathematics and the five conventions; this story makes that
  contract binding on `Cosmos/Source` and `CombustionInstability`, enforced by a
  double-cover property test rather than by a PDF.
