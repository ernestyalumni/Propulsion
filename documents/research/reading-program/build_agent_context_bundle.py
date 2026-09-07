#!/usr/bin/env python3
"""Build a `<Slug>-AgentContext` bundle for a parsed book.

The ReadingRoom server (`ReadingRoom/server.py`) does not read the corpus
directly; it reads these bundles out of the exports directory. Until now they
were assembled by hand, which is why adding a book to the reading room was a
chore rather than a command. This does it from the parsed corpus and the
reading-program directory, and writes the `MANIFEST.json` that
`tools/verify_manifest.py` checks.

What the ReadingRoom actually requires from a bundle (everything else is
onboarding material for an agent reading the package):

    context/reading-program/<slug>/progress.json   book metadata + chapter list
    context/reading-program/<slug>/ROADMAP.md      linked from the book card
    context/reading-program/<slug>/READING-LEDGER.md
    corpus/Public/books/<subject>/<slug>/toc.json  flat list with pdf_page
    corpus/Public/books/<subject>/<slug>/INDEX.md
    corpus/Public/books/<subject>/<slug>/chapters/NNN-*.md
    corpus/Public/books/<subject>/<pdf_stem>.pdf   the original, beside the slug

Usage:
    build_agent_context_bundle.py SLUG \
        [--corpus DIR] [--reading-program DIR] [--template BUNDLE] [--out DIR]
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import date

# The chunk-level marker output is intermediate: large, and reproducible from
# the PDF. Everything a reader or an agent needs is in the assembled products.
EXCLUDE_DIRS = {"chunks", "__pycache__", "nougat_out.bak"}
EXCLUDE_SUFFIXES = {".pyc", ".bak"}


def copy_tree(source, destination, exclude_dirs=EXCLUDE_DIRS):
    for root, dirs, files in os.walk(source):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        target_root = os.path.join(destination, os.path.relpath(root, source))
        os.makedirs(target_root, exist_ok=True)
        for name in files:
            if any(name.endswith(s) for s in EXCLUDE_SUFFIXES):
                continue
            shutil.copy2(os.path.join(root, name), os.path.join(target_root, name))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("slug")
    parser.add_argument("--corpus", default="/media/propdev/Expansion/openclaw/"
                        ".openclaw/workspace/Data/Public/books/EngineeringPhysics")
    parser.add_argument("--reading-program",
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--template", default="/media/propdev/Expansion/openclaw/"
                        ".openclaw/workspace/Data/Exports/ForPropulsion/"
                        "Sutton-RocketPropulsionElements-9e-AgentContext",
                        help="existing bundle to take the shared context and tools from")
    parser.add_argument("--out", default="/media/propdev/Expansion/openclaw/"
                        ".openclaw/workspace/Data/Exports/ForPropulsion")
    args = parser.parse_args()

    slug_dir = os.path.join(args.corpus, args.slug)
    spec = json.load(open(os.path.join(slug_dir, "book_spec.json")))
    subject = os.path.basename(args.corpus)
    package = f"{args.slug}-AgentContext"
    bundle = os.path.join(args.out, package)

    if os.path.exists(bundle):
        shutil.rmtree(bundle)
    os.makedirs(bundle)

    # 1. Shared context and tools, taken from the template bundle.
    for relative in ("context/PROPULSION_CHARTER.md", "context/PROJECT_CONTEXT.md"):
        source = os.path.join(args.template, relative)
        if os.path.exists(source):
            os.makedirs(os.path.dirname(os.path.join(bundle, relative)), exist_ok=True)
            shutil.copy2(source, os.path.join(bundle, relative))
    for relative in ("context/propulsion-stories", "tools"):
        source = os.path.join(args.template, relative)
        if os.path.isdir(source):
            copy_tree(source, os.path.join(bundle, relative))

    # 2. Reading-program material: the shared README plus this book's directory.
    program_readme = os.path.join(args.reading_program, "README.md")
    if os.path.exists(program_readme):
        write_to = os.path.join(bundle, "context/reading-program/README.md")
        os.makedirs(os.path.dirname(write_to), exist_ok=True)
        shutil.copy2(program_readme, write_to)
    book_program = os.path.join(args.reading_program, args.slug)
    if not os.path.isdir(book_program):
        sys.exit(f"no reading-program directory for {args.slug} at {book_program}")
    copy_tree(book_program, os.path.join(bundle, "context/reading-program", args.slug))

    # 3. The corpus subtree and the original source document beside it.
    corpus_books = os.path.join(bundle, "corpus/Public/books", subject)
    copy_tree(slug_dir, os.path.join(corpus_books, args.slug))
    source_pdf = os.path.join(args.corpus, spec["source_pdf"])
    if not os.path.isfile(source_pdf):
        sys.exit(f"source PDF missing: {source_pdf}")
    os.makedirs(corpus_books, exist_ok=True)
    shutil.copy2(source_pdf, os.path.join(corpus_books, os.path.basename(source_pdf)))

    # 4. Package documentation.
    authors = " and ".join(spec["authors"])
    today = date.today().isoformat()
    write(os.path.join(bundle, "README.md"), f"""# {package}

This is an offline onboarding package for **{authors} — {spec['title']}, {spec.get('edition', '')}**, created {today}.

Start with `AGENTS.md`, then the corpus `INDEX.md`, then the matching reading roadmap in `context/reading-program/`. The original source PDF is included under the corpus path.

The archive is a snapshot, not a live synchronization mechanism. Its `MANIFEST.json` checks every payload file. To validate after extraction:

```bash
python3 tools/verify_manifest.py {package}
```

All paths below `corpus/` are relative to a receiving machine's corpus root. Keep the directory layout when importing.

Do not treat OCR output as ground truth for a disputed equation: use the supplied source document, exact page map, and reconciliation records. This book is a **scan**, so that warning binds harder here than for a born-digital text — every character in the corpus came from OCR, and there is no text layer to fall back on.
""")

    write(os.path.join(bundle, "AGENTS.md"), f"""# Agent instructions — {package}

1. Read `README.md`, then `context/PARSE_AND_PROVENANCE.md`, before relying on the corpus.
2. Use the corpus `INDEX.md` and `page_map.json` for citations. Cite title, edition, section, printed page, and PDF page. For this book, **PDF page = printed page + 12** in the body; the front matter carries roman folios equal to the PDF page.
3. Treat `chapters/`, `pages/`, and `book.md` as navigation-friendly working text; use the original source document to settle OCR or equation ambiguity. `artifacts/equations.json` carries the book's own equation numbers and is the right place to look one up.
4. Read `context/PROPULSION_CHARTER.md` and `context/propulsion-stories/README.md` before proposing changes to the Propulsion project. Stories 12-14 are research sketches, not authorization to implement.
5. **Read this book with Sutton.** `context/reading-program/{args.slug}/PAIRING-SUTTON.md` is the row-by-row pairing and the joint reading order. Hill and Peterson derive what Sutton states; taking either alone loses the point.
6. The source book is private copyrighted material. Do not redistribute this archive or the source text; do not commit any of it.

This package is read-only reference material. Put new work in the receiving project, never back into `corpus/`.
""")

    write(os.path.join(bundle, "INSTALL_AND_VERIFY.md"), f"""# Install and verify

1. Copy the archive to the receiving machine without changing it.
2. Extract it: `unzip {package}-{today}.zip`
3. Verify before use: `python3 tools/verify_manifest.py {package}`
4. Move or copy `corpus/Public/books/{subject}/` into that machine's configured corpus root, preserving the names below it. Do not import into a repository `Data/` fixture directory.
5. If an existing corpus has a complete parse record for the same source, compare manifests and keep the existing corpus rather than overwriting it.

The manifest deliberately omits itself from its file list because a self-checksum is recursive.
""")

    template_license = os.path.join(args.template, "LICENSE_AND_USE.md")
    if os.path.exists(template_license):
        shutil.copy2(template_license, os.path.join(bundle, "LICENSE_AND_USE.md"))

    parse_record = {
        "source_document": "300 dpi bilevel scan, CCITT G4, no text layer",
        "marker_extraction": "complete (marker_book.py, page-accurate, force_ocr)",
        "nougat_extraction": spec.get("parse_state", {}).get("nougat", "see MANIFEST"),
        "page_map": "folio rule hand-verified against the scanned running heads, "
                    "then re-checked against every chapter heading "
                    "(page_map_verification.json)",
        "chapter_split_and_locator_index": "complete, split by exact PDF page range",
        "artifact_inventory": "equations, tables and figures extracted from the "
                              "Marker block tree with the book's own numbers",
        "normalization": "Cyrillic homoglyphs repaired (see scripts/textnorm.py); "
                         "Greek deliberately untouched, it is real mathematics here",
    }
    write(os.path.join(bundle, "context/PARSE_AND_PROVENANCE.md"),
          "# Parse and provenance\n\n```json\n"
          + json.dumps(parse_record, indent=2)
          + "\n```\n\nThe package includes the exact source document, the full parsed "
            "corpus subtree, the assembled Markdown, per-page and per-chapter splits, "
            "the equation/table/figure inventories, and the locator metadata. Logs are "
            "provenance, not proof that an OCR string is correct.\n\n"
            "**This book is a scan.** Unlike the born-digital texts in this program, "
            "there is no text layer behind the OCR. Check any equation that matters "
            "against the PDF page named in `page_map.json`.\n")

    # 5. Manifest last, over everything written above.
    files = []
    total = 0
    for root, dirs, names in os.walk(bundle):
        dirs[:] = sorted(dirs)
        for name in sorted(names):
            path = os.path.join(root, name)
            relative = os.path.relpath(path, bundle)
            if relative == "MANIFEST.json":
                continue
            size = os.path.getsize(path)
            total += size
            files.append({"path": relative, "size_bytes": size, "sha256": sha256(path)})

    write(os.path.join(bundle, "MANIFEST.json"), json.dumps({
        "schema_version": 2,
        "package_name": package,
        "created_at": today,
        "portability": ["Ubuntu/Linux", "macOS"],
        "corpus_root_in_archive": "corpus",
        "original_source_documents_present": True,
        "source_selection": "Complete current parsed corpus subtree plus its "
                            "original source document; the intermediate Marker "
                            "chunk output is omitted as reproducible.",
        "parse_record": parse_record,
        "manifest_self_policy": "MANIFEST.json is excluded from its own file list "
                                "because self-hashing is recursive.",
        "file_count": len(files),
        "payload_size_bytes": total,
        "files": files,
    }, indent=2))

    print(f"{package}: {len(files)} files, {total / 1e6:.1f} MB")
    print(f"-> {bundle}")
    print(f"verify: python3 {bundle}/tools/verify_manifest.py {package}")


if __name__ == "__main__":
    main()
