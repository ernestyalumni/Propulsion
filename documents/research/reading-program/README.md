# Reading program — three books, one library

The endeavor: read the classical texts that underpin spacecraft simulation,
GNC, and propulsion, and rewrite their methods from the physics and the
equations into a multi-physics library that is Rust first, C++ where the
existing stack needs it, and CUDA where the work is data-parallel. Stories 15
and 16 in `documents/stories/` are the binding requirements; the seven-step
first-principles protocol is in
`../numerical-recipes-rewrite/ROADMAP.md` §2 and applies to every book.

| Book | Slug (parsed corpus) | Roadmap | Ledger | Parse state |
|---|---|---|---|---|
| Press et al., Numerical Recipes 3e | `NumericalRecipes-3e` (under `Data/Public/books/Physics/Press …/`) | [`../numerical-recipes-rewrite/ROADMAP.md`](../numerical-recipes-rewrite/ROADMAP.md) | [`../numerical-recipes-rewrite/READING-LEDGER.md`](../numerical-recipes-rewrite/READING-LEDGER.md) | Marker + Nougat + reconcile done 2026-08-24; vision pass partial |
| Wie, Space Vehicle Dynamics and Control 2e | `Wie-SpaceVehicleDynamicsControl-2e` (under `Data/Public/books/EngineeringPhysics/`) | [`Wie-SpaceVehicleDynamicsControl-2e/ROADMAP.md`](Wie-SpaceVehicleDynamicsControl-2e/ROADMAP.md) | [`Wie-SpaceVehicleDynamicsControl-2e/READING-LEDGER.md`](Wie-SpaceVehicleDynamicsControl-2e/READING-LEDGER.md) | Marker + Nougat + reconcile + chapter split done 2026-09-02 (agree 944, conflict 338, marker-only 360, nougat-only 131); vision pass pending |
| Sutton and Biblarz, Rocket Propulsion Elements 9e | `Sutton-RocketPropulsionElements-9e` (same directory) | [`Sutton-RocketPropulsionElements-9e/ROADMAP.md`](Sutton-RocketPropulsionElements-9e/ROADMAP.md) | [`Sutton-RocketPropulsionElements-9e/READING-LEDGER.md`](Sutton-RocketPropulsionElements-9e/READING-LEDGER.md) | batch OCR started 2026-09-02 |

## The dashboard

`dashboard/` holds a generator that turns the three `progress.json` files, the
ledgers, the git log, and the Rust test count into one HTML page, published as
an artifact so it can be opened from any device. To refresh it, any agent runs:

```bash
cd documents/research/reading-program/dashboard
python3 generate_dashboard.py            # writes dashboard.html next to it
# then republish dashboard.html to the URL recorded in DASHBOARD-URL.md
```

Progress is recorded in three places and the dashboard only reads them:

1. `<book>/progress.json` — one entry per chapter: rank, why, target module,
   language, and `status` ∈ {`todo`, `reading`, `read`, `module`, `read-only`}.
   An agent that finishes a chapter's seven artifacts flips `status` to
   `module` and names the module.
2. `<book>/READING-LEDGER.md` — one row per section actually read to the
   standard.
3. The code itself: `Cosmos/Rust` (`cargo test`), `Cosmos/Source` (gtest).

## Parsed corpus layout (what an agent can open today)

Each slug directory under `Data/Public/books/EngineeringPhysics/` holds:

- `INDEX.md` — section → printed page → PDF page; grep it.
- `toc.json`, `page_map.json` — the same, machine-readable.
- `ocr-compare/<stem>.marker.md` — the Marker backbone (complete coverage).
- `ocr-compare/nougat_out/<stem>.mmd` — the Nougat equation pass.
- `ocr-compare/reconciled/` — `merged.md`, `equations.json`, and after the
  vision pass `equations_resolved.json` and `resolved.md`.
- `chapters/NNN-slug.md` + `INDEX.md` — the Marker backbone split into the
  book's real chapters by `split_by_toc.py` using `toc.json` (the older
  `split_chapters.py` guessed from heading levels and fragmented Wie into 205
  parts; `run_batch_ocr.sh` now prefers the TOC split when `toc.json` exists).
  Wie: 16 parts (front matter, 14 chapters, index). Sutton: front matter,
  21 chapters, 3 appendices, index.
- `<slug>.tex` — pandoc LaTeX of the backbone, math preserved.

Batch driver: `Data/Public/books/EngineeringPhysics/run_batch_ocr.sh PHASE [slug …]`
reading `BOOKS.tsv`; every phase is resumable. Log of the 2026-09-02 run:
`batch-ocr-2026-09-02.log` in that directory.

## Source-file decisions, 2026-09-02

- **Wie 2e.** Two PDFs were present. `Space Vehicle Dynamics and Control by
  Wie, Bong (z-lib.org)(1).pdf` (863 pp) begins at printed page 121 and lacks
  Chapter 1 entirely. `4.860119.pdf` (970 pp, AIAA, DOI 10.2514/4.860119) is
  complete: Chapter 1 at printed 3–120, the original index at 935, supporting
  materials at 951. `BOOKS.tsv` now points at `4.860119.pdf`. The AIAA
  watermark leaked into only 15 lines of the Marker backbone (stripped in
  place; original kept as `4.860119.marker.md.bak-prefilter`) and into none of
  the Nougat output or the crop-mark strings, so the z-lib copy's one
  advantage was moot; **it was deleted on 2026-09-02** per Ernest's
  conditional instruction.
- **Sutton equation tags.** The book numbers equations `(2-1)`, hyphenated
  by chapter. Marker emits `\tag{2-1}`; Nougat emits `\tag{2.1}` or drops the
  chapter prefix entirely. The first reconcile aligned nothing (0 agree, 39
  nougat-only); the Marker tags were normalized to `2.1` in
  `ocr-compare/normalized/` and the reconcile rerun (agree 6, conflict 2,
  marker-only 119, nougat-only 31; the raw result is kept in
  `reconciled.raw-tags/`). Treat Sutton as a Marker-backbone book, the way
  Arnold is recorded in story 01.
- **Sutton 9e.** `Rocket Propulsion Elements ( PDFDrive ).pdf` is the 2017
  Wiley 9th edition (title page verified). A 10th edition exists and should
  replace it when a digital copy is obtained; the roadmap is written against
  9e section numbers and will need re-anchoring.
