# Session handoff: Hill & Peterson parse, the reading room, and force as a 1-form

**Prepared:** 2026-09-08
**Repositories:** `Propulsion` (branch `feat/reading-room-companions`),
`Monoclaw` (branch `feat/scanned-book-ocr-pipeline`)
**Maturity:** tooling committed and pushed; one topic note written; several
decisions still open
**Audience:** any agent — Claude, Grok, Codex, Hermes. Nothing here depends on
a particular assistant's private memory.

## 1. Orientation in one paragraph

Ernest is reading Sutton & Biblarz *Rocket Propulsion Elements* 9e alongside
Hill & Peterson *Mechanics and Thermodynamics of Propulsion* 2e, on the thesis
that Sutton states results and Hill & Peterson derive them. This session
parsed Hill & Peterson (a 766-page **scan**, which needed a new page-accurate
OCR pipeline), parsed Frankel *The Geometry of Physics*, wired Hill & Peterson
into the reading-room app as a fourth book, and wrote the first
mathematical-physics topic note: force as a one-form, and what that makes
impulse. He is currently reading Sutton Chapter 2.

## 2. Where things live

The corpus is **outside** every repository, per
[`../../stories/02-corpus-root.md`](../../stories/02-corpus-root.md). Resolve
it from `PROPULSION_CORPUS_ROOT`; never hardcode an absolute path.

| What | Path |
|---|---|
| Corpus root | `$PROPULSION_CORPUS_ROOT` (currently `<workspace>/Data`) |
| Parsed books | `<CORPUS_ROOT>/Public/books/{EngineeringPhysics,Physics}/<slug>/` |
| AgentContext bundles | `<CORPUS_ROOT>/Exports/ForPropulsion/<slug>-AgentContext/` |
| Reading-room live state | `<CORPUS_ROOT>/ReadingRoom/propulsion/progress.json` |
| Built note PDFs | `<CORPUS_ROOT>/ReadingRoom/propulsion/artifacts/` (8 files) |
| OCR tooling | `Monoclaw/Python/ocr-compare/scripts/` |

Three things are deliberately outside git, for three different reasons: the
**book corpus** (copyright, size, and story 02), the **built PDFs**
(regenerable; `notes.py:outside_repo()` raises if you try), and the **live
reading state** (personal, high-churn; `server.py` errors if `--state-dir` is
inside the repo). Sources are versioned; outputs and corpus are not.

## 3. Git state — read this before assuming anything is unsaved

**Monoclaw**, branch `feat/scanned-book-ocr-pipeline`, pushed:
`778d0fb` (pipeline) and `29e49eb` (force_ocr flag).

> **Gotcha.** The Monoclaw *working checkout* sits on an unrelated branch
> (`feat/sage-agent-launcher`, a SageMath effort). The ocr-compare files
> therefore appear **untracked** in `git status` there even though they are
> committed and pushed on the other branch. They were committed via
> `git worktree`, deliberately, to avoid entangling the two efforts. Verify
> with `git show origin/feat/scanned-book-ocr-pipeline:<path>` before
> concluding work is lost.

**Propulsion**, branch `feat/reading-room-companions`, pushed: `9d8eb66`
(Hill & Peterson in the reading program).

Uncommitted in Propulsion, and *why*:

- `documents/notes/` — **entirely untracked**, including this session's
  `topics/force-one-form.tex`. Nothing in `.gitignore` excludes it; it is
  simply part of an in-progress "notes companion" feature that predates this
  session. These `.tex` sources *should* be versioned; they just have not been
  committed yet.
- `ReadingRoom/{notes.py,build_notes.py,resources.json,web/notes.*,...}`,
  `Studies/` — the same in-progress notes feature. **Not this session's work.**
  It was deliberately left alone rather than swept into a commit by an agent
  that did not write it and cannot vouch for it.
- `documents/research/reading-program/build_agent_context_bundle.py` — this
  session's corpus-root fix (see §7), not yet committed.
- `ReadingRoom/{server.py,web/app.js,tests/test_server.py}` — these carry
  *both* the notes-feature WIP and this session's Hill & Peterson wiring. The
  Hill & Peterson hunks were committed surgically in `9d8eb66`; what remains
  uncommitted in those files is the notes feature only.

## 4. What was parsed

| Book | Slug | Pages | Equations | Notes |
|---|---|---:|---:|---|
| Hill & Peterson 2e | `HillPeterson-MechanicsThermodynamicsPropulsion-2e` | 766 | 1523 | 300 dpi bilevel **scan**, no text layer; 88 tables, 470 figures |
| Frankel, Geometry of Physics | `Frankel-GeometryOfPhysics-3e` | 749 | 2809 | born-digital; parsed with `--no-force-ocr` |

Hill & Peterson quality evidence: 766/766 pages, 0 failed chunks, folio rule
(**PDF = printed + 12** in the body; roman folio = PDF page in front matter)
verified against **24/24** chapter headings, and only 17 Cyrillic homoglyphs
needing repair. Frankel needed **0** homoglyph repairs, which is the signature
of a text layer passing through untouched.

Two duplicate Hill & Peterson PDFs in the books directory are **byte-identical**
(MD5 `e712cd95f56bdbd04563dc65764618be`) — one file under two names; either is
redundant.

## 5. The reading room

```sh
cd <Propulsion>            # server picks up SPECS from ReadingRoom/server.py
python3 -B ReadingRoom/server.py            # http://127.0.0.1:8876
python3 -B ReadingRoom/build_notes.py --output <CORPUS_ROOT>/ReadingRoom/propulsion/artifacts
```

Four books load with zero warnings (`nr`, `wie`, `sutton`, `hp`); 14/14 tests
pass (`python3 -m unittest tests.test_server`). `build_notes.py` produces 8
PDFs and is strict — it aborts on any `undefined`, `multiply defined` or
`Overfull` in any log, and publishes atomically only if all 8 pass.

Recorded reading progress: **Sutton Chapter 1 complete** (§1, 1.1, 1.2, 1.3
marked read, with a note on §1 that it is a qualitative landscape chapter,
consistent with its rank-18 `read-only` placement).

Dashboard artifact (4 books):
<https://claude.ai/code/artifact/4b1c3653-ba81-431c-8ebf-70fb60bb5bfd> —
regenerate with `dashboard/generate_dashboard.py`, republish to that same URL.

## 6. The topic note

`documents/notes/topics/force-one-form.tex`, in the master and the Sutton
companion. Argument: work is a canonical pairing needing no metric, so force
is naturally a covector; momentum is one for the same reason (Legendre
transform); impulse inherits it. Consequences developed there, each stated as
a proposition with hypotheses:

- `I_t = ∫F dt` is one *component* `J(ê)` of an impulse covector, and total
  impulse is only well defined at all because the problem is flat with a
  global parallel frame.
- `‖J‖ ≤ ∫‖f‖dt` with equality iff thrust direction is constant — a quoted
  total impulse is an **upper bound** for any gimbaled or slewing burn.
- Sutton Eq. 2-3 is the **mass-flow-weighted** mean of instantaneous `I_s`,
  not a time average.
- `c = F/ṁ` silently applies the musical isomorphism: `c ê = (f/ṁ)^♯`.

**Erratum found and verified** against the source PDF's own text layer (not
OCR): Sutton Example 2-1, printed p. 31, prints `F = I_s g₀ = 23.3 × 2354.4 =
54,936 N`. `I_s g₀` is a *velocity* (`c`), not a force; the correct form is
`F = ṁc`. Separately, `23.3 × 2354.4 = 54,857.5`, not the printed 54,936 — the
printed figure comes from the unrounded `ṁ = 70/3`. All downstream numbers are
correct; only the symbolic statement is wrong, in the 9th edition.

**Symbol convention adopted.** Sutton's `c` (effective exhaust velocity)
collides with Hill & Peterson's `c` (absolute fluid velocity, `c_r,c_θ,c_z`),
*not* with speed of sound — both books use `a`, as does `nozzle.tex`. Rule:
`c` for exhaust velocity in rocket-performance context, subscripted `c_z,c_θ`
for turbomachinery per H&P, `v_e` where both appear together.

## 7. Open decisions — these need Ernest, not an agent

1. **Commit `documents/notes/`?** The `.tex` sources are unversioned. Doing so
   means deciding what to do with the co-resident notes-feature WIP.
2. **The bibliography change.** `master.tex` and `companions/sutton.tex` now
   place `\bibliography` *outside* `notesbody`; `wie.tex` and
   `numerical-recipes.tex` still have it inside. This was necessary — multicol
   balancing the bibliography produced a 19 pt overfull once the document grew
   (`\raggedcolumns` and `\raggedbottom` both failed to fix it). The four
   documents are now structurally inconsistent though all 8 PDFs build. Unify
   or revert?
3. **Schematic topology extraction.** Prototype done for Sutton ch. 1–2 only
   (`Sutton-.../artifacts/schematics/`: 10 mermaid + JSON topologies from 18
   figures classified). It does **not** scale as a script — each topology is a
   vision-reading judgement per figure. Extending to all of Sutton is ~80–150
   more figures. Decide chapter-by-chapter against what is actually being read.

## 8. Natural next steps

- **Sutton §2.2 (Thrust).** Ernest is reading Chapter 2 now, and the note
  deliberately stops short of deriving `F = ṁv_e + (p_e − p_a)A_e`, flagging it
  as a control-volume consequence rather than a primitive. That is the obvious
  next topic, and it is where the covector-valued traction 2-form earns its
  keep.
- **Hill & Peterson nougat output is unreconciled.** `ocr-compare/nougat_out/`
  holds a completed cross-check pass, but `combine.py` assumes the older flat
  layout, not the page-accurate one. Marker is the backbone and is verified;
  nougat here is a spot-check resource only.
- **Frankel is parsed but unregistered** — no `book_spec.json`, not in
  `BOOKS.tsv`, not in the reading program. It is usable as `book.md` /
  `pages/` / `artifacts/` but has no folio rule or TOC anchoring.

## 9. Gotchas that cost time here

- **Marker's JSON renderer cannot be `model_dump()`ed.** Its image dict is
  keyed by `BlockId` objects and picture blocks carry base64 payloads. Serialize
  the block tree by hand (`block_to_dict`).
- **Marker strips running heads**, leaving `PageHeader` blocks empty, so the
  printed folio cannot be recovered from OCR. It comes from a hand-verified
  folio rule in `book_spec.json`, which `build_page_map.py` then re-checks
  against every chapter heading.
- **Three page numberings.** Marker's `{K}` separator is 0-based; filenames and
  anchors are 1-based PDF pages; citations use the printed folio.
  `page_map.json` resolves between them. Confusing them is the standard bug.
- **`force_ocr` is wrong for born-digital PDFs.** Default is on (for scans);
  pass `--no-force-ocr` otherwise or you discard a good text layer.
- **`pgrep -f <script>` matches its own shell wrapper**, so it reports a
  finished job as still running. Confirm with `ps aux | grep "[m]arker"`.
