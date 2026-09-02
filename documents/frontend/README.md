# Front end — saved, versioned, and openable without an account

Every page this project publishes exists in three forms. The point of this
directory is that **the last one needs nothing but a browser**: no claude.ai,
no login, no network.

| Form | Where | Needs |
|---|---|---|
| Published artifact | the URLs in `pages.json` | a claude.ai login (they are private to Ernest's account) |
| Artifact body | the `source` path in `pages.json` | nothing; but it is a fragment, not a document |
| **Standalone page** | **`site/`** | **a browser** |

## Open it now

```bash
xdg-open documents/frontend/site/index.html
```

Or open `file:///…/Propulsion/documents/frontend/site/index.html` in any
browser, on this machine or any machine with a clone of the repository.
`site/index.html` links to every page.

Web fonts come from Google Fonts when the machine is online and fall back to
system serif, sans, and monospace faces when it is not. Nothing else is
fetched; the pages carry no scripts and no tracking. Every other URL in them
is an ordinary hyperlink you choose to click.

## Why the sources are fragments

The Artifact tool publishes a *body*: the file carries no `<!doctype>`,
`<html>`, `<head>`, or `<body>`, because claude.ai supplies those at publish
time along with the title and the emoji favicon. Opened directly, such a file
renders but has no title, no favicon, and no declared charset.
`wrap_standalone.py` supplies exactly what the publisher would have, so the
standalone copy is the same page rather than an approximation.

## Rebuild

```bash
documents/frontend/build.sh
```

That regenerates the reading board from repository state, wraps each body into
a standalone document, and rewrites `site/index.html`. Run it after any change
worth seeing, then republish the bodies to their artifact URLs if you want the
hosted copies to match. Never edit anything under `site/` by hand; it is
generated.

## Versioning

Git is the version history: the bodies, the standalone pages, and the
generators are all committed, so any past state of a page can be recovered
with `git log -- documents/frontend/site`. claude.ai keeps its own version
history for the published copies, but that history is not a backup of this
repository and this repository is not a mirror of it. The two are kept in
step by rerunning `build.sh` and republishing.

## Adding a page

1. Write the artifact body somewhere sensible in `documents/`.
2. Append an entry to `pages.json`: `file`, `title`, `emoji`, `description`,
   `source`, `artifact` URL, and whether it is `generated`.
3. Add two lines to `build.sh` wrapping it into `site/`.
4. Run `build.sh`.

## Sharing beyond this machine

The `site/` directory is plain static files with no build step and no server
requirement, so any of these work:

- **Another of your machines:** clone the repository and open `site/index.html`.
- **Phone or tablet on the same network:** `python3 -m http.server 8000 -d documents/frontend/site`
  and open `http://<this-machine-ip>:8000` from the device.
- **A real URL:** copy `site/` to any static host. It is self-contained.
- **claude.ai:** republish the bodies; the links in `pages.json` stay stable.

## Pages

See `pages.json` for the machine-readable list. Today:

- **Cosmos Reading Board** (📚) — progress across all three books, generated
  from `progress.json` files, ledgers, corpus directories, `cargo test`,
  the gtest binary, and `git log`.
- **Numerical Recipes for Spacecraft, Propulsion, and GNC** (🛰️) — the reading
  guide: five arcs, every chapter ranked, the language policy, the rewrite
  queue, and the bibliography of substitutes. Hand-written, not generated.
