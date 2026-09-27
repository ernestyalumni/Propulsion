# Propulsion Reading Room

A local study workspace for Numerical Recipes 3e, Wie 2e, Sutton 9e, Hill & Peterson 2e,
Huzel & Huang (1992) and Humble (1995). The roadmap page maps where the four
propulsion books overlap (`documents/research/reading-program/FOUR-BOOK-OVERLAP.json`).
The Huzel & Huang and Humble bundles are built by
`documents/research/reading-program/build_reading_room_bundle.py`.
The bookshelf, linked roadmap, original PDF reader, section notes, and existing
simulation labs work offline after the browser dependencies are installed.

The **Notes & solutions** collection links book sections to shared LaTeX
derivations, portrait/screen PDFs, SageMath checks, numerical implementations,
and dated evidence. Each book card and reader section links to related work.
Search by topic or manuscript; follow a locator to return to the book. Missing
optional repositories are labeled unavailable. Existing personal progress is
independent of scientific verification.

## Build the companions and run their studies

From the repository root, choose an absolute artifacts directory outside Git:

```sh
python3 -B ReadingRoom/build_notes.py --output /absolute/path/to/artifacts
python3 -B Studies/verify.py --output /absolute/path/to/artifacts/verification.json --sage-image YOUR_INSTALLED_IMAGE_ID
python3 -B ReadingRoom/server.py --artifacts-dir /absolute/path/to/artifacts --open-browser
```

The default artifacts directory is `workspace/Data/ReadingRoom/propulsion/artifacts`.
There are three complete starter studies (oscillator, rigid body, nozzle), plus
a foundations chapter and a legacy manuscript index. All four documents share
topic sources. Eight PDFs are built with citations checked and LaTeX shell
execution disabled. Sage runs in an existing prebuilt image with no pull,
source build or container network. Omit `--sage-image` for numerical-only
evidence, which is labeled accordingly. See [the studies](../Studies/README.md)
and [document authoring](../documents/notes/README.md).

For local model-assisted development, see [PDD with llama.cpp](docs/LOCAL_PDD.md).
`bash ReadingRoom/pdd-local.sh ...` (from the repo root) uses the saved localhost
configuration without provider keys. It is separate from the reader UI.

PDF build hashes and computation input hashes distinguish current artifacts
from stale ones. Evidence is a reproducible local record, not an independent
review certificate. Historical notes and code do not inherit verification
from the starter studies. No browser endpoint executes code or edits LaTeX.

Optional references are discovered as sibling repositories, then under the
same workspace's `workspace2/repos/`. Override any root explicitly:

```sh
python3 -B ReadingRoom/server.py --reference-root mathphysics=/path/to/mathphysics --reference-root Monoclaw=/path/to/Monoclaw --reference-root CompPhys=/path/to/CompPhys
```

Only individual files registered in `resources.json` are served from those
roots; symlinks escaping their resolved roots are rejected.

## Open it

On this Mac, double-click `Open Reading Room.command`, or from the repo root:

```sh
python3 -B ReadingRoom/server.py --open-browser
```

Visit **http://127.0.0.1:8876**. Keep the terminal running; Ctrl+C stops it.
The server binds only to loopback. It is a personal local tool, not a deployment
server. Run only one server against a given progress directory.

First-time setup on another checkout (Node/npm and Python 3.9+):

```sh
cd ReadingRoom
npm ci --ignore-scripts --omit=optional --no-audit --no-fund
python3 -B server.py --open-browser
```

Dependencies are pinned in `package-lock.json`. PDF.js (Apache-2.0), KaTeX,
Marked, and DOMPurify (MIT; DOMPurify also offers Apache-2.0) are served from
local `node_modules`, never a CDN. Package licenses remain in their installed
directories. No build or Python packages are required.

The reader uses PDF.js's bundled `legacy/build` compatibility entry points so
Chrome 144 can open PDFs without the newer `Map.getOrInsertComputed` API.
This does not change the pinned package or add browser network requests.

## Reading together

- Start/resume a book, search the contents, or choose a section from the roadmap.
- The page box always means **PDF page**. Section headings show the printed page
  and mark estimated PDF mappings with `≈`. Arrow keys also change PDF pages.
- Zoom and within-page vertical scroll are saved with the PDF bookmark.
- Parsed text opens the chapter for Wie/Sutton, or the current extracted PDF page
  for Numerical Recipes. It retains the PDF bookmark. Math is rendered where the
  source has LaTeX; source OCR conflicts remain unresolved by this app.
- Notes belong to the selected section shown above them. Paging within the PDF
  does not change that section. Choose another contents entry to switch note scope.
- Reading, discussion, derivation, and implementation are independent self-recorded
  checks. Opening a page records position, not completion. The historical exported
  ledger is available separately and is never counted as fresh local progress.
- Notes autosave after a short typing pause; **Save notes** saves immediately.
  If saving fails or another tab has changed progress, the page keeps your draft,
  reports the problem, and blocks internal navigation. Copy the draft before
  reloading after a conflict. Closing with unsaved work triggers a browser warning.
- **Copy session context** prepares a handoff for our conversation. **Export session**
  downloads the on-disk handoff. There is no embedded AI chat or automatic access
  to your browser position by an agent.
- The quaternion lab is linked as an existing interactive visualization. Nozzle
  theory and the PI step-size controller are source-code links, not new runnable
  simulations or claims that their tests have passed.

## Data and agent handoff

Default input: `workspace/Data/Exports/ForPropulsion/*-AgentContext/`.
Default output: `workspace/Data/ReadingRoom/propulsion/`:

- `progress.json`: canonical versioned state, bookmarks, per-section notes/checks.
- `HANDOFF.md`: readable summary regenerated when progress is saved.

An agent resuming the study should read those two output files, then the selected
bundle's `AGENTS.md`, provenance, and section index. Use the source PDF to resolve
equation ambiguity. Historical code status in the bundles may belong to another
machine; verify local availability separately. The original bundles remain intact.

Paths are inferred from this repository's workspace location, not the obsolete
Linux paths in exported `progress.json`. Override them when needed:

```sh
python3 -B ReadingRoom/server.py \
  --exports /absolute/path/to/ForPropulsion \
  --state-dir /absolute/path/to/reading-progress \
  --port 8876
```

State must be outside the repository and exported bundles. JSON saves use atomic
replacement and revision checks to reject stale browser writes. The handoff is
derived; `/api/handoff` can regenerate it from canonical state if needed. Back up
the progress directory with your normal local backups. Corrupt canonical JSON
stops startup instead of discarding previous progress.

Only configured book documents, specific application assets, and the existing
quaternion lab are exposed. Supplied Numerical Recipes code distributions are
not served. No files or notes are uploaded. Changing the port changes the URL,
but not the saved progress.

## Validation

```sh
cd ReadingRoom
python3 -B -m unittest discover -s tests -v
node tests/browser.cjs
```

The Python tests read these three real bundles and write only temporary progress.
The Chrome suite starts its own temporary server and state directory, restarts
the server to verify persistence, exercises all books and parsed math, checks
mobile layout and conflict handling, and rejects unexpected external requests.
It uses the workspace's existing Playwright installation; elsewhere set
`PLAYWRIGHT_MODULE` to an installed Playwright module and install Chrome.

These checks validate the reader, not the mathematical correctness of the books,
OCR, or linked simulation implementations. `PRODUCT_INTENT.md` records the
accepted first-version scope and source ownership.
