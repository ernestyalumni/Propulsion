# First-version validation — 2026-09-06

Branch: `feat/local-reading-room`. Conventional source; no PDD generation or
semantic/LLM verification claimed.

- `python3 -B -m unittest discover -s tests -v`: **14 tests passed**.
  Tests cover all three real bundle indexes, mapping certainty, PDF ranges,
  persistent notes/bookmarks/handoff, stale-tab conflicts, malformed progress,
  cross-origin/token rejection, arbitrary-path and symlink rejection, corrupt
  state preservation, disk-write failure, and immutable source snapshots/PDFs.
- `node tests/browser.cjs`: **passed in headless Google Chrome**. All three PDFs
  render with selectable text; Wie/Sutton parsed math and NR page text render;
  page/scroll/zoom and notes survive a full server restart; notes remain scoped
  to sections; immediate export includes the just-typed draft; roadmap links,
  historical ledger, and existing quaternion visualization open; a stale tab
  preserves its unsaved draft; desktop and 390px layouts have no page overflow;
  no JavaScript page errors or external resource requests occurred.
- Desktop bookshelf, reader, roadmap, and mobile screenshots visually inspected.
- JavaScript syntax, launcher shell syntax, and `git diff --check` passed.
- Original orbit-demo working-tree edit preserved. No existing physics source
  changed; their broader C++/Rust/Python test suites were not rerun.

Tests use temporary progress, separate from the user's live progress directory.
Browser test screenshots stay in the temporary directory printed by the test,
not in Git; reader screenshots may contain pages from the local books.

Known first-version limits: no embedded AI chat, PDF annotation editing, arbitrary
book import UI, or new simulations. Parsed-text scroll is not a separate bookmark;
switching views retains the original PDF bookmark. User-marked “implemented” is
not automatically inferred from tests. Only one server should write a given state
directory; browser tabs are protected by revision conflicts.
