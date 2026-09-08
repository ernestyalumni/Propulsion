# Reading Room companions — implementation handoff

Branch: `feat/reading-room-companions`, based on Propulsion `1976675`.
Canonical checkout: `/media/propdev/Expansion/openclaw/.openclaw/workspace/repos/Propulsion`.
Ernest requested consolidation into this original repository and deletion of
the temporary clone. The earlier read-only diagnosis confused the sandbox's
mount view with the host filesystem, which is writable with authorized access.
No mount repair, commit or push was performed. File-transfer checks are retained
in the main workspace's `reviews/repository-consolidation-2026-09-06/`.

## Delivered

- Notes & solutions collection, book/section links, filtering/search, return to
  source sections, registered source/PDF/code/evidence links and missing-asset states.
- Shared LaTeX foundations, oscillator, rigid-body, nozzle and historical-source
  map; thematic master plus three companions, each in portrait and screen editions.
- Independent numerical teaching implementations and exact SageMath checks,
  dated metrics/version/image/hash records, and stale-evidence/build detection.
- Eight-PDF builder with BibTeX, shell escape disabled, references/layout checked,
  intermediate outputs outside the repository, and build provenance.
- Existing prebuilt Sage image for offline batch verification; optional
  authenticated, localhost-only Jupyter launcher with separate scratch storage.
- The original reading/bookmark/notes behavior and private corpus boundary.
- PDF.js's pinned compatibility build for Chrome 144, which lacks
  Map.getOrInsertComputed required by the package's default build.

## Local artifacts and execution

The three 2026-09-02 archives were extracted without overwriting existing files
under `/home/propdev/.openclaw/workspace/Data/Exports/ForPropulsion/`.
These private book bundles were not copied into the repository.

Built PDFs and verification/build records:
`/home/propdev/.openclaw/workspace/Data/ReadingRoom/propulsion/artifacts/`.
Start from the receiving checkout with `python3 -B ReadingRoom/server.py`.
Default URL: `http://127.0.0.1:8876/#notes`.

On this Linux machine the original workspace's `Data/Exports/ForPropulsion`
and `Data/ReadingRoom/propulsion` link to these existing main-workspace data
directories. Books, built PDFs and reading progress stay outside Git and are
not duplicated or reset by checkout consolidation.

The three new studies have explicit chosen synthetic inputs and finite
verification scope. Existing large manuscripts were indexed and linked, not
silently merged or declared mathematically correct. The documents are working
companions with complete starter examples, not complete solutions to all three books.

## Evidence

- Original baseline: 14 backend tests passed before feature changes.
- Expanded backend: 21 tests cover original behavior, actual registered HTTP
  access, locators, unavailable roots, traversal/symlink protections, corrupt
  evidence, source changes invalidating verification, and PDF replacement hashes.
- Full Chrome acceptance: original PDFs and parsed math, restart persistence,
  zoom/scroll, notes, conflicts, roadmap/lab, five topics, companion collection,
  source links, search, section round trip, 390px layout, no external requests.
- All eight LaTeX builds pass with resolved citations and no overfull boxes;
  screen PDF visually inspected. Master: five portrait pages, three screen pages.
- Sage 10.8, installed image
  `sha256:aeef59a8c17212357aeb83bd7f1cfac43bb1c789ae3fa2d887227c763bdfcac3`.
  All three symbolic and numerical checks pass, with no pull/build/network.
- Oscillator refinement ratios: about 4.00125 and 4.00031; relative energy error
  bounded by 0.000625 over 10,000 steps. Rigid-body refinement ratio: about 16;
  inertial momentum error about 6.17e-10. Nozzle Mach roots: about 0.239543 and
  2.442765, with normalized mass-flux residual below 1e-12.
- Jupyter smoke: unauthenticated sessions API returns 403 on isolated localhost
  port 18889, then the test session/container is stopped.

## PDD status and local skill

**Later same-day update:** the fork is now at `e02419065` on
`feat/local-only-no-github-auth`. Key-free llama.cpp inference was exercised
successfully on Ernest's Linux CUDA server and configured for this Reading
Room. There are now 27 passing backend tests. See [LOCAL_PDD.md](LOCAL_PDD.md)
for accepted/rejected model outputs, the new CLI exit-status bug and explicit
agentic limitations. The earlier steering failure described below is fixed;
115 focused PDD tests pass. The earlier provider attempt below remains history,
not a successful apply. The local skill has been updated and revalidated.

Ernest approved the structure; read-only plan ID `take-a-look-here-ca36e009`.
`intent apply` recorded the exact request in `docs/intents/`, then provider
startup failed. Automatic approval review rejected the escalated retry because
it could transmit private request/repository context to an external provider
without explicit authorization. No external-provider bypass was attempted.
The interrupted runtime record has an explicit agent annotation; it is not a
successful PDD report.

Implementation and bounded specifications were prepared directly in this coding
session. No provider-generated architecture, story contract, semantic validation
or completed PDD sync is claimed. `PRODUCT_INTENT.md` retains existing acceptance
and the approved extension; `docs/PRODUCT_INTENT.md` is the PDD intake record.

PDD fork inspected at master `4989fbddb`: 69 targeted tests passed, one local
steering test failed. `drain_issue_steers` checks for gh before reaching its
local comment route. Auth guards pass, but this is not a fully working mid-run
local steering replacement. Details and local:1 are in the workspace audit
`reviews/pdd-local-workflows-2026-09-06/`. No PDD fork files were changed.

Local skill `/home/propdev/.agents/skills/pdd/SKILL.md` now defaults to
`PDD_LOCAL_ONLY=1`, documents the local tracker, distinguishes provider network
from GitHub auth, preserves approval across turns, and records the tested
steering limitation. Skill validation passed.
