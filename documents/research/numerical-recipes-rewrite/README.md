# Numerical Recipes rewrite — research directory

The endeavor: read Numerical Recipes 3e cover to cover in relevance order, and
rewrite each method from the physics and the equations into a multi-physics
simulation library that is Rust first, C++ where the existing stack needs it,
and CUDA where the work is data-parallel.

Read in this order:

1. [ROADMAP.md](ROADMAP.md) — language policy, the seven-step first-principles
   protocol, every chapter ranked with the physics that earns its place, and
   the sequence for the next ten sessions.
2. [spacex-signal-2026-09-01.md](spacex-signal-2026-09-01.md) — the dated
   capability record (story 06 form) that set the ranking.
3. [READING-LEDGER.md](READING-LEDGER.md) — one row per section actually read
   to the standard; nothing counts as read until its note, module, tests, and
   sidecar exist.
4. The reading guide artifact (five arcs, chapter ranking, bibliography of
   substitutes): https://claude.ai/code/artifact/3a63d17a-ec6f-4f40-b6f0-f9c93113cb31
5. Stories 15 and 16 in `documents/stories/` — the binding requirements.

Where the work lives:

- `Cosmos/Rust/` — the Rust workspace (`cosmos_numerical`), golden vectors, and
  the C++ emitter tools. `cargo test` runs everything.
- `Cosmos/Source/Numerical/` — the C++ that already exists (chapter 17 done).
- `documents/derivations/` — one first-principles note per method.
- Parsed book: `Data/Public/books/Physics/Press …/NumericalRecipes-3e/`
  (`sections.json`, `equation_index.json`, `READING_GUIDE.md`; PDF page =
  printed page + 24; do not re-parse).
