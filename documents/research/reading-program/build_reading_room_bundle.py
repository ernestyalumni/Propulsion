#!/usr/bin/env python3
"""Build a reading-room `<Slug>-AgentContext` bundle for Huzel & Huang, Humble,
Williams or Turns.

`build_agent_context_bundle.py` copies a whole parsed corpus, which suits the
Sutton and Hill & Peterson corpora. The Huzel & Huang and Humble corpora carry
gigabytes of review evidence (renders, OCR checkpoints, alignment features)
that the reading room never reads. This builds only what `ReadingRoom/server.py`
requires:

    context/reading-program/<slug>/{progress.json,ROADMAP.md,READING-LEDGER.md}
    corpus/Public/books/<subject>/<slug>/toc.json         section-level, server format
    corpus/Public/books/<subject>/<slug>/INDEX.md
    corpus/Public/books/<subject>/<slug>/chapters/NNN-*.md
    corpus/Public/books/<subject>/<pdf_stem>.pdf

The section contents come from each corpus's source-checked contents record
where one exists: Humble `curated/contents-index.json`; Huzel & Huang the
seven transcribed contents pages plus `reference/downloaded-page-map.json`
(printed folio -> downloaded PDF page). Williams and Turns have chapter-level
contents only (`toc.json`), so their sections come from the parsed headings
(`parsed/headings.json`), placed by `curated/folio-rules.json`: Williams's
numbered sections (7.8, 9.1.4.7, ...), Turns's unnumbered all-caps headings.

Unnumbered Huzel & Huang and Turns sections get the id `<parent>~<slug>`,
never an invented section number; their displayed number stays empty.

Williams and Turns chapter files are built by joining the reconciled
(source-checked) page transcriptions, because their `parsed/chapters/*.md`
are page link lists.

Usage:
    PROPULSION_CORPUS_ROOT=<CORPUS_ROOT> build_reading_room_bundle.py {huzel,humble,williams,turns}
"""

import argparse
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

BOOKS = {
    "humble": {
        "slug": "Humble-SpacePropulsionAnalysisDesign",
        "pdf": "Space Propulsion Analysis and Design by Ronald W. Humble (z-lib.org).pdf",
    },
    "huzel": {
        "slug": "HuzelHuang-ModernEngineeringLiquidPropellantRocketEngines-1992",
        "pdf": "vdoc.pub_modern-engineering-for-design-of-liquid-propellant-rocket-engines.pdf",
    },
    "williams": {
        "slug": "Williams-CombustionTheory-2e",
        "pdf": "[Forman_A._Williams]_Combustion_Theory(BookSee.org).pdf",
    },
    "turns": {
        "slug": "Turns-IntroductionToCombustion-3e",
        "pdf": "dokumen.pub_an-introduction-to-combustion-concepts-and-applications-3rd-ed-978-0-07-338019-3-0-07-338019-9.pdf",
    },
}

SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "vs", "with"}
# Turns back-matter headings that are not reading-program entries.
TURNS_SKIP = {"OVERVIEW", "SUMMARY", "NOMENCLATURE", "REFERENCES", "REFERENCE", "PROBLEMS", "PROJECTS",
              "REVIEW QUESTIONS", "QUESTIONS AND PROBLEMS", "PROBLEMS AND PROJECTS"}


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def depth_of(number):
    return number.count(".") + 1


def humble_toc(corpus):
    entries = json.load(open(os.path.join(corpus, "curated/contents-index.json")))["entries"]
    chapters = {c["chapter"]: c["title"] for c in json.load(open(os.path.join(corpus, "toc.json")))}
    toc = []
    for e in entries:
        section = e["section"].strip()
        if not section or section == "Index":
            continue
        number = section.replace("App. ", "")
        is_chapter = number in chapters or number in "ABC"
        title = chapters.get(number, e["title"]) if is_chapter else e["title"]
        if number in "ABC":
            title = e["title"]
        toc.append({"number": number, "title": title,
                     "printed_page": int(e["resolved_printed_reference"]), "pdf_page": e["pdf_page"],
                     "depth": 1 if is_chapter else depth_of(number), "pdf_page_exact": True,
                     "kind": "chapter" if is_chapter else "section"})
    return toc


def huzel_toc(corpus):
    page_map = json.load(open(os.path.join(corpus, "reference/downloaded-page-map.json")))
    to_pdf = {}
    for p in page_map["pages"]:
        if p["printed_page"] is not None and not p["duplicate"]:
            to_pdf.setdefault(p["printed_page"], p["canonical_downloaded_pdf_page"])

    def locate(printed):
        if printed in to_pdf:
            return to_pdf[printed], True
        below = max(k for k in to_pdf if k < printed)
        return to_pdf[below] + (printed - below), False

    rows = []
    for i in range(1, 8):
        text = open(os.path.join(corpus, f"master/pages/front-contents-{i}.md")).read()
        for line in text.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 3 or not cells[2].isdigit():
                continue
            rows.append(cells)

    pages = [int(r[2]) for r in rows]
    # A contents misprint shows as one page above both neighbours (e.g. "341"
    # printed for a subsection between pp. 234 and 242 in 7.2).
    misprints = {i for i in range(1, len(pages) - 1)
                 if pages[i] > pages[i + 1] and pages[i] > pages[i - 1]}
    toc, notes = [], []
    chapter, parent, last_page = None, None, 0
    for index, (label, title, page) in enumerate(rows):
        page = int(page)
        title = re.sub(r"\$\s*\(?([^$]*?)\)?\s*\$", lambda m: m.group(1), title).replace("\\", "").strip()
        m = re.match(r"(Chapter|Appendix)\s+(\w+)", label)
        if m:
            chapter = parent = m.group(2)
            number, depth, kind, ident = chapter, 1, "chapter", None
        elif label:
            parent = number = label
            depth, kind, ident = 2, "section", None
        elif chapter is None:
            continue  # foreword and preface: not reading-program entries
        elif title.startswith(("List of", "Subject Index")):
            continue
        else:
            number, depth, kind = "", 3, "subsection"
            ident = f"{parent}~{slugify(title)}"
        exact_note = None
        if index in misprints:
            # Keep the entry, place it at the preceding entry's page, mark it approximate.
            exact_note = f"contents prints p. {page}; out of sequence, placed at p. {last_page}"
            notes.append(f"{title}: {exact_note}")
            page = last_page
        elif page < last_page:
            sys.exit(f"contents out of order at {title!r}: p. {page} after p. {last_page}")
        pdf, exact = locate(page)
        row = {"number": number, "title": title, "printed_page": page, "pdf_page": pdf,
               "depth": depth, "pdf_page_exact": exact and exact_note is None, "kind": kind}
        if ident:
            row["id"] = ident
        if depth > 1:
            row["parent"] = chapter
        toc.append(row)
        last_page = page
    ids = [r.get("id") or r["number"] for r in toc]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        sys.exit(f"duplicate section ids: {sorted(duplicates)}")
    return toc, notes


def title_case(text):
    """Title-case an all-caps heading; keep chemical formulas and roman numerals."""
    words = []
    for i, word in enumerate(text.split()):
        lower = word.lower()
        if any(c.isdigit() for c in word) or re.fullmatch(r"[IVX]+", word):
            words.append(word)
        elif i and lower in SMALL_WORDS:
            words.append(lower)
        else:
            words.append("-".join(part[:1].upper() + part[1:].lower() for part in word.split("-")))
    return " ".join(words)


def clean_heading(text):
    text = re.sub(r"\\[a-zA-Z]+", "", text).replace("{", "").replace("}", "").replace("$", "")
    text = re.sub(r"\s+", " ", text).replace("Simplifi ed", "Simplified").strip().rstrip(".")
    return title_case(text) if text.isupper() else text


def printed_folio(corpus):
    """PDF page -> printed folio (int), from curated/folio-rules.json; None off the arabic body."""
    rules = [r for r in json.load(open(os.path.join(corpus, "curated/folio-rules.json")))["rules"]
             if r.get("style") != "roman" and "blank" not in r["printed_equals"]]

    def folio(pdf):
        rule = next((r for r in rules if r["pdf_from"] <= pdf <= r["pdf_to"]), None)
        return pdf - rule["offset"] if rule else None
    return folio


def heading_toc(book, corpus):
    """Chapters from toc.json plus sections from parsed/headings.json (Williams, Turns)."""
    folio = printed_folio(corpus)
    chapters = json.load(open(os.path.join(corpus, "toc.json")))
    headings = json.load(open(os.path.join(corpus, "parsed/headings.json")))
    toc, seen = [], set()
    for ch in chapters:
        number = ch["chapter"]
        if not re.fullmatch(r"\d+|[A-F]", number):
            continue  # indexes
        toc.append({"number": number, "title": ch["title"], "printed_page": ch["printed_start"],
                    "pdf_page": ch["pdf_start"], "depth": 1, "pdf_page_exact": True, "kind": "chapter"})
        seen.add(number)
        for h in headings:
            pdf = h["pdf_page"]
            if not ch["pdf_start"] <= pdf <= ch["pdf_end"] or folio(pdf) is None:
                continue
            raw = h["title"].strip()
            row = {"printed_page": folio(pdf), "pdf_page": pdf, "pdf_page_exact": True, "parent": number}
            if book == "williams":
                m = re.match(r"((?:\d+|[A-E])(?:\.\d+)+)\.?\s+(.+)", raw)
                if not m or m.group(1).split(".")[0] != number:
                    continue
                row.update(number=m.group(1), title=clean_heading(m.group(2)),
                           depth=m.group(1).count(".") + 1, kind="section")
                ident = row["number"]
            else:
                if not raw.isupper() or len(raw) < 4 or raw in TURNS_SKIP or raw.startswith(("APPENDIX", "AN INTRODUCTION")):
                    continue
                title = clean_heading(raw)
                ident = f"{number}~{slugify(title)}"
                row.update(number="", title=title, depth=2, kind="section", id=ident)
            if ident in seen:
                continue
            seen.add(ident)
            toc.append(row)
    return toc


def page_chapter_texts(corpus):
    """Chapter file name -> Markdown joined from the reconciled page transcriptions."""
    pages = os.path.join(corpus, "parsed/reconciled-pages")
    texts = {}
    for ch in json.load(open(os.path.join(corpus, "toc.json"))):
        if not ch["chapter"].isdigit():
            continue
        parts = []
        for pdf in range(ch["pdf_start"], ch["pdf_end"] + 1):
            path = os.path.join(pages, f"{pdf:04d}.md")
            if not os.path.exists(path):
                continue
            lines = [line for line in open(path).read().splitlines()
                     if not line.startswith(("[Source image]", "Reading transcription with"))]
            parts.append("\n".join(lines).strip())
        texts[f"{int(ch['chapter']):03d}-{slugify(ch['title'])}.md"] = (
            f"# {ch['chapter']}. {ch['title']}\n\n" + "\n\n".join(parts) + "\n")
    return texts


def chapter_files(book, corpus):
    """Map chapter number -> source Markdown, named NNN-*.md for the reader."""
    if book == "humble":
        folder = os.path.join(corpus, "parsed/chapters")
        pattern = r"(\d+)-(.+)\.md"
    else:
        folder = os.path.join(corpus, "master/chapters")
        pattern = r"(\d+)-(.+)\.md"
    result = {}
    for name in os.listdir(folder):
        m = re.fullmatch(pattern, name)
        if m:
            result[f"{int(m.group(1)):03d}-{m.group(2)}.md"] = os.path.join(folder, name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("book", choices=sorted(BOOKS))
    parser.add_argument("--corpus-root", default=os.environ.get("PROPULSION_CORPUS_ROOT"))
    parser.add_argument("--subject", default="EngineeringPhysics")
    args = parser.parse_args()
    if not args.corpus_root:
        parser.error("PROPULSION_CORPUS_ROOT is not set; export it or pass --corpus-root")
    root = os.path.abspath(args.corpus_root)
    spec = BOOKS[args.book]
    slug = spec["slug"]
    books_dir = os.path.join(root, "Public/books", args.subject)
    corpus = os.path.join(books_dir, slug)
    pdf = os.path.join(books_dir, spec["pdf"])
    program = os.path.join(HERE, slug)
    for label, path in (("corpus", corpus), ("source PDF", pdf), ("reading program", program)):
        if not os.path.exists(path):
            sys.exit(f"{label} missing: {path}")

    if args.book == "humble":
        toc, notes = humble_toc(corpus), []
    elif args.book == "huzel":
        toc, notes = huzel_toc(corpus)
    else:
        toc, notes = heading_toc(args.book, corpus), []

    bundle = os.path.join(root, "Exports/ForPropulsion", f"{slug}-AgentContext")
    staging = bundle + ".partial"
    if os.path.exists(staging):
        shutil.rmtree(staging)
    target = os.path.join(staging, "corpus/Public/books", args.subject, slug)
    os.makedirs(os.path.join(target, "chapters"))
    shutil.copytree(program, os.path.join(staging, "context/reading-program", slug))
    with open(os.path.join(target, "toc.json"), "w") as f:
        json.dump(toc, f, indent=1, ensure_ascii=False)
    shutil.copy2(os.path.join(corpus, "INDEX.md"), os.path.join(target, "INDEX.md"))
    if args.book in ("williams", "turns"):
        for name, text in page_chapter_texts(corpus).items():
            with open(os.path.join(target, "chapters", name), "w") as f:
                f.write(text)
    else:
        for name, source in chapter_files(args.book, corpus).items():
            shutil.copy2(source, os.path.join(target, "chapters", name))
    shutil.copy2(pdf, os.path.join(staging, "corpus/Public/books", args.subject, spec["pdf"]))
    with open(os.path.join(staging, "README.md"), "w") as f:
        f.write(f"# {slug}-AgentContext\n\nReading-room bundle built by "
                f"`documents/research/reading-program/build_reading_room_bundle.py`. "
                f"It holds only what the reading room reads; the full parsed corpus "
                f"and its review evidence stay in `Public/books/{args.subject}/{slug}/`.\n"
                + "".join(f"\nContents note: {n}\n" for n in notes))
    if os.path.exists(bundle):
        shutil.rmtree(bundle)
    os.replace(staging, bundle)
    print(f"{bundle}: {len(toc)} contents entries"
          + (f"; {len(notes)} contents note(s)" if notes else ""))


if __name__ == "__main__":
    main()
