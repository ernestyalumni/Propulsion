# Reading program — four books, one library

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
| Hill and Peterson, Mechanics and Thermodynamics of Propulsion 2e | `HillPeterson-MechanicsThermodynamicsPropulsion-2e` (same directory) | [`HillPeterson-MechanicsThermodynamicsPropulsion-2e/ROADMAP.md`](HillPeterson-MechanicsThermodynamicsPropulsion-2e/ROADMAP.md) | [`HillPeterson-MechanicsThermodynamicsPropulsion-2e/READING-LEDGER.md`](HillPeterson-MechanicsThermodynamicsPropulsion-2e/READING-LEDGER.md) | scanned-source pipeline, 2026-09-06 |

## Read Sutton with Hill and Peterson

Sutton and Hill & Peterson are **one reading track, not two**. Sutton states
the result and gives the design data; Hill and Peterson derive it from a
control volume, so you can see which assumptions were spent. The row-by-row
pairing table, the joint reading order, and the note on where the two books
disagree are in
[`HillPeterson-MechanicsThermodynamicsPropulsion-2e/PAIRING-SUTTON.md`](HillPeterson-MechanicsThermodynamicsPropulsion-2e/PAIRING-SUTTON.md).
Start there before opening either book's roadmap. The short version: take
H&P Ch. 3 before any nozzle work, and H&P Ch. 4 before Sutton's Bartz
correlation in §8.5.

## The dashboard

`dashboard/` holds a generator that turns the four `progress.json` files, the
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

### Scanned books: the page-accurate pipeline (added 2026-09-06)

Hill and Peterson is a **scan** — 766 pages of 300 dpi bilevel images with no
text layer at all, unlike every other book here. `marker_chunked.py`
concatenates markdown into one flat blob, which is enough for a born-digital
PDF and not enough for a scan: with no text layer there is nothing to fall
back on when a page is missed, and the figures exist only as pixels. Four
tools in `Monoclaw/Python/ocr-compare/scripts/` were added for it, and they
apply to any future scanned book:

- `marker_book.py` — runs Marker in resumable chunks with `paginate_output`,
  `force_ocr`, and the **document built once, rendered twice** (Markdown and
  JSON block tree) so the structured output costs no extra GPU time. Emits
  per-chunk markdown and JSON, extracted figure crops, and a `.ok` sentinel
  per chunk so a crash at page 700 does not discard pages 0–699.
- `assemble_book.py` — chunks → `pages/page-NNNN.md` (one file per PDF page),
  one `book.md`, and `page_index.json` reporting missing, duplicate, and
  suspiciously thin pages.
- `build_page_map.py` — expands a verified `book_spec.json` into
  `page_map.json` and `toc.json`, then **verifies** the folio rule by looking
  for each chapter heading where the rule predicts it.
- `extract_artifacts.py` — walks the block tree into `equations`, `tables`,
  and `figures` (JSON + Markdown), keeping the book's own numbers
  (`Eq. (4.7a)`, `FIGURE 4.10`, `TABLE 10.8`) as the join key.

A scanned book therefore carries three page numberings, and mixing them up is
the easy mistake: Marker's 0-based page id, the 1-based PDF page (what a
viewer shows, and what every tool here addresses), and the folio printed on
the paper (what a citation uses). `page_map.json` is the resolver.

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
- **Hill and Peterson 2e (2026-09-06).** Two files in the books directory,
  `Hill Peterson 1992 Mechanics and thermodynamics of propulsion.pdf` and
  `[Hill_Peterson]_Mechanics_and_thermodynamics_of_pr(b-ok.cc).pdf`, are
  **byte-identical** — same size and the same MD5
  `e712cd95f56bdbd04563dc65764618be`. They are one file under two names, not
  two scans to choose between; either can be deleted without loss. `BOOKS.tsv`
  and `book_spec.json` name the first. The scan is 766 pages, 300 dpi, CCITT
  G4 bilevel, produced by Pixel Translations PIXPDF in 2002, with **no text
  layer** (11 characters across pages 30–40). Page geometry is clean and
  square, so Marker's OCR does well on it; the folio rule (PDF = printed + 12
  in the body, roman folio = PDF page in the front matter) was verified by
  reading the scanned running heads at PDF pages 105, 130, 251, 380, 520, 663,
  755, and 759.

- **Sutton 9e.** `Rocket Propulsion Elements ( PDFDrive ).pdf` is the 2017
  Wiley 9th edition (title page verified). A 10th edition exists and should
  replace it when a digital copy is obtained; the roadmap is written against
  9e section numbers and will need re-anchoring.
