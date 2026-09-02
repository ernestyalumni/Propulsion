# Cosmos Reading Board

Live page: https://claude.ai/code/artifact/4b1c3653-ba81-431c-8ebf-70fb60bb5bfd
(private to Ernest's account; share from the page's menu if needed).

First published 2026-09-02 from commit `0f90437` on `feat/numerical-recipes-rust-rewrite`.

## Refresh protocol (any agent, any session)

1. `python3 generate_dashboard.py` in this directory. It reads the three
   `progress.json` files, the reading ledgers, the parsed-corpus directories
   (OCR phase detection by file presence and log freshness), runs
   `cargo test` in `Cosmos/Rust` and the gtest binary in `Cosmos/BuildGcc`,
   and reads `git log`. Output: `dashboard.html` beside the script.
2. Republish `dashboard.html` with the Artifact tool, passing the URL above
   as `url` so the same page updates. Never pass a favicon on a redeploy.
3. Do not edit `dashboard.html` by hand; edit the `progress.json` files, the
   ledgers, or the generator.

## What to change when work happens

- Finished a chapter's seven artifacts: set that chapter's `status` to
  `module` in its book's `progress.json`, name the module, and add the
  ledger row. Started one: `reading`.
- New Rust module: nothing; the generator scans `Cosmos/Rust/cosmos_numerical/src`.
- OCR phases: nothing; detected from the corpus directories.
