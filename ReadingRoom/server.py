#!/usr/bin/env python3
"""Loopback-only reading room. Python standard library; corpus is read-only."""
import argparse
import copy
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit, parse_qs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKSPACE = REPO.parent.parent
SPECS = [
    ("nr", "NumericalRecipes3e-AgentContext", "NumericalRecipes-3e", "17.1", "Numerical methods", "24"),
    ("wie", "Wie-SpaceVehicleDynamicsControl-2e-AgentContext", "Wie-SpaceVehicleDynamicsControl-2e", "5.4", "Dynamics & control", "18"),
    ("sutton", "Sutton-RocketPropulsionElements-9e-AgentContext", "Sutton-RocketPropulsionElements-9e", "3.3", "Rocket propulsion", "24"),
]
LABS = [
    {"id": "quaternion", "title": "Quaternion Convention Lab", "kind": "Interactive lab", "book": "wie", "section": "5.4", "path": "Cosmos/QuaternionConventionLab/web/index.html", "description": "Explore q versus −q, scalar layout, and active/passive rotations.", "url": "/lab/Cosmos/QuaternionConventionLab/web/index.html"},
    {"id": "nozzle", "title": "Nozzle theory", "kind": "Source code", "book": "sutton", "section": "3.3", "path": "NozzleTheory.py", "description": "Existing symbolic nozzle relations. A starting point for our next experiment.", "url": "/source/nozzle"},
    {"id": "integrator", "title": "Adaptive Runge–Kutta", "kind": "Source code", "book": "nr", "section": "17.2", "path": "Cosmos/Source/Numerical/ODE/RKMethods/ComputePIStepSize.h", "description": "Inspect the existing C++ PI step-size controller alongside the text.", "url": "/source/integrator"},
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def contained(root, path):
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


class Catalog:
    def __init__(self, exports):
        self.exports = exports.resolve()
        self.books = {}
        self.files = {}
        self.text_pages = {}
        self.warnings = []
        for ident, folder, slug, start, domain, offset in SPECS:
            try:
                self.load_book(ident, folder, slug, start, domain, int(offset))
            except (OSError, ValueError, KeyError, StopIteration) as exc:
                self.warnings.append({"book": ident, "message": "Bundle unavailable or invalid: " + str(exc)})

    def load_book(self, ident, folder, slug, start, domain, offset):
        bundle = contained(self.exports, self.exports / folder)
        progress_file = next((bundle / "context/reading-program").glob("*/progress.json"))
        progress = read_json(contained(bundle, progress_file))
        corpus = next((bundle / "corpus/Public/books").glob("*/" + slug), None)
        if ident == "nr":
            corpus = next((bundle / "corpus/Public/books/Physics").glob("*/NumericalRecipes-3e"))
        corpus = contained(bundle, corpus)
        pdf = contained(bundle, corpus.parent / (progress["pdf_stem"] + ".pdf"))
        if not pdf.is_file():
            raise ValueError("Original PDF missing")
        raw = read_json(corpus / ("sections.json" if ident == "nr" else "toc.json"))
        if isinstance(raw, dict):
            raw = [dict(value, number=key, pdf_page_exact=True) for key, value in raw.items()]
        sections = []
        for i, row in enumerate(raw):
            if not isinstance(row.get("pdf_page"), int):
                continue
            sections.append({"id": row.get("number") or "entry-" + str(i), "number": row.get("number", ""),
                             "title": row["title"], "printed_page": row["printed_page"], "pdf_page": row["pdf_page"],
                             "exact": row.get("pdf_page_exact", False), "depth": row.get("depth", 2)})
        paths = {"pdf": pdf, "roadmap": progress_file.parent / "ROADMAP.md",
                 "ledger": progress_file.parent / "READING-LEDGER.md", "index": corpus / "INDEX.md"}
        chapters = []
        for ch in progress["chapters"]:
            chapter = dict(ch)
            chapter["historical_status"] = chapter.pop("status", "todo")
            chapter["historical_notes"] = chapter.pop("notes", "")
            chapters.append(chapter)
        for p in (corpus / "chapters").glob("*.md"):
            if re.match(r"\d{3}-", p.name):
                paths["chapter-" + p.name[:3]] = p
        if ident == "nr":
            text = contained(bundle, corpus.parent / "parsed/Numerical.Recipes.3ed.txt")
            self.text_pages[ident] = text.read_text(encoding="utf-8").split("\f")
        self.files[ident] = {key: contained(bundle, path) for key, path in paths.items() if path.is_file()}
        self.books[ident] = {"id": ident, "title": progress["title"], "short": progress["short"], "domain": domain,
                             "authors": progress["authors"], "publisher": progress["publisher"], "pages": progress["pages"],
                             "offset": offset, "start": start, "sections": sections, "chapters": chapters,
                             "snapshot": "2026-09-02", "pdf_path": str(pdf),
                             "text_chapters": [key for key in self.files[ident] if key.startswith("chapter-")],
                             "ocr_note": "Parsed text is a reading aid. Check equations against the original PDF; OCR adjudication is incomplete."}

    def section(self, book, section):
        return next((s for s in self.books[book]["sections"] if s["id"] == section), None)


class StateStore:
    def __init__(self, directory, catalog):
        self.directory = directory.resolve()
        self.path = self.directory / "progress.json"
        self.catalog = catalog
        self.lock = threading.Lock()
        self.read()  # Fail visibly on corrupt state, before accepting any writes.

    def read(self):
        if not self.path.exists():
            return {"schema": 1, "revision": 0, "last_book": None, "books": {}}
        result = read_json(self.path)
        if result.get("schema") != 1 or type(result.get("revision")) is not int or not isinstance(result.get("books"), dict):
            raise ValueError("Invalid saved reading state; preserve it and repair before restarting")
        return result

    def atomic_write(self, path, text):
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".reading-", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def update(self, payload):
        book = payload["book"]
        if book not in self.catalog.books:
            raise ValueError("Unknown book")
        patch = payload["patch"]
        if not isinstance(patch, dict) or set(patch) - {"bookmark", "section"}:
            raise ValueError("Unsupported progress field")
        with self.lock:
            state = self.read()
            if payload.get("revision") != state["revision"]:
                raise Conflict("Progress changed in another tab. Reload after copying your unsaved notes.")
            entry = copy.deepcopy(state["books"].get(book, {"sections": {}}))
            if "bookmark" in patch:
                mark = patch["bookmark"]
                if not isinstance(mark, dict) or set(mark) != {"page", "section", "zoom", "scroll"}:
                    raise ValueError("Invalid bookmark")
                if type(mark["page"]) is not int or not 1 <= mark["page"] <= self.catalog.books[book]["pages"]:
                    raise ValueError("PDF page out of range")
                if not self.catalog.section(book, mark["section"]):
                    raise ValueError("Unknown section")
                for key, lo, hi in [("zoom", 0.5, 2.5), ("scroll", 0, 1)]:
                    if type(mark[key]) not in (int, float) or not lo <= mark[key] <= hi:
                        raise ValueError("Invalid " + key)
                entry["bookmark"] = mark
            if "section" in patch:
                section = patch["section"]
                if not isinstance(section, dict) or set(section) != {"id", "notes", "questions", "next", "checks"}:
                    raise ValueError("Invalid section record")
                if not self.catalog.section(book, section["id"]):
                    raise ValueError("Unknown section")
                for field in ("notes", "questions", "next"):
                    if not isinstance(section[field], str) or len(section[field]) > 40000:
                        raise ValueError("Note is too long or invalid")
                if not isinstance(section["checks"], dict) or set(section["checks"]) != {"read", "discussed", "derived", "implemented"}:
                    raise ValueError("Invalid learning checks")
                if any(type(v) is not bool for v in section["checks"].values()):
                    raise ValueError("Learning checks must be boolean")
                entry.setdefault("sections", {})[section["id"]] = section
            entry["updated_at"] = datetime.now(timezone.utc).isoformat()
            state["books"][book] = entry
            state["last_book"] = book
            state["revision"] += 1
            self.atomic_write(self.path, json.dumps(state, indent=2, ensure_ascii=False) + "\n")
            # JSON is canonical; the handoff is derived and can always be regenerated.
            warning = None
            try:
                self.atomic_write(self.directory / "HANDOFF.md", self.handoff(state))
            except OSError:
                warning = "Progress saved, but HANDOFF.md could not be refreshed. Use Export session."
            return {"state": state, "warning": warning}

    def handoff(self, state=None):
        state = state or self.read()
        lines = ["# Propulsion reading session", "", "Canonical state: " + str(self.path),
                 "", "Checks are self-recorded, not automated validation. Exported 2026-09-02 progress remains historical.", ""]
        for ident, data in state["books"].items():
            if ident not in self.catalog.books:
                continue
            book = self.catalog.books[ident]
            lines += ["## " + book["title"], "", "Source PDF: " + book["pdf_path"]]
            mark = data.get("bookmark")
            if mark:
                sec = self.catalog.section(ident, mark["section"])
                lines += ["Resume: section " + mark["section"] + " — " + sec["title"],
                          "PDF page: " + str(mark["page"]) + " (section starts at printed page " + str(sec["printed_page"]) + ")"]
            for sid, row in data.get("sections", {}).items():
                lines += ["", "### Section " + sid, "Checks: " + ", ".join(k for k, v in row["checks"].items() if v)]
                for key, title in [("notes", "Notes"), ("questions", "Open questions"), ("next", "Next experiment")]:
                    if row.get(key):
                        lines += ["", title + ":", row[key]]
            lines += [""]
        return "\n".join(lines) + "\n"


class Conflict(ValueError):
    pass


class ReadingServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, exports, state_dir):
        self.catalog = Catalog(exports)
        self.store = StateStore(state_dir, self.catalog)
        self.token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Do not log book paths, notes, tokens, or request payloads.

    def trusted(self):
        port = self.server.server_address[1]
        hosts = {"127.0.0.1:" + str(port), "localhost:" + str(port)}
        if self.headers.get("Host") not in hosts:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {"http://" + h for h in hosts}

    def respond(self, data, content_type="application/json; charset=utf-8", status=200, extra=None):
        if isinstance(data, (dict, list)):
            data = json.dumps(data, ensure_ascii=False)
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def file(self, path):
        if not path.is_file():
            raise FileNotFoundError()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix in (".js", ".mjs"):
            content_type = "text/javascript"
        if path.suffix in (".md", ".py", ".h", ".txt"):
            content_type = "text/plain; charset=utf-8"
        data = path.read_bytes()
        byte_range = self.headers.get("Range")
        if byte_range:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", byte_range)
            if not match:
                self.respond("Invalid range", "text/plain", 416)
                return
            start = int(match[1])
            end = min(int(match[2]) if match[2] else len(data) - 1, len(data) - 1)
            if start > end:
                self.respond("Invalid range", "text/plain", 416, {"Content-Range": "bytes */" + str(len(data))})
                return
            self.respond(data[start:end + 1], content_type, 206, {"Content-Range": "bytes %d-%d/%d" % (start, end, len(data)), "Accept-Ranges": "bytes"})
            return
        self.respond(data, content_type, extra={"Accept-Ranges": "bytes"})

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self.trusted():
            self.respond({"error": "Local origin required"}, status=403)
            return
        url = urlsplit(self.path)
        path = unquote(url.path)
        parts = path.strip("/").split("/")
        try:
            if path == "/api/bootstrap":
                self.respond({"books": list(self.server.catalog.books.values()), "warnings": self.server.catalog.warnings,
                              "state": self.server.store.read(), "token": self.server.token,
                              "state_path": str(self.server.store.path),
                              "labs": [dict(lab, available=(REPO / lab["path"]).is_file()) for lab in LABS]})
            elif path == "/api/handoff":
                self.respond(self.server.store.handoff(), "text/markdown; charset=utf-8")
            elif len(parts) == 3 and parts[0] == "book":
                self.file(self.server.catalog.files[parts[1]][parts[2]])
            elif len(parts) == 3 and parts[:2] == ["api", "text"]:
                pages = self.server.catalog.text_pages[parts[2]]
                page = int(parse_qs(url.query).get("page", [1])[0])
                if not 1 <= page <= len(pages):
                    raise ValueError("Page out of range")
                self.respond(pages[page - 1], "text/plain; charset=utf-8")
            elif len(parts) == 2 and parts[0] == "source":
                lab = next(x for x in LABS if x["id"] == parts[1])
                self.file(contained(REPO, REPO / lab["path"]))
            elif parts[0] == "lab":
                candidate = contained(REPO, REPO.joinpath(*parts[1:]))
                allowed = [REPO / "Cosmos/QuaternionConventionLab/web", REPO / "Cosmos/Source/Examples/Astrodynamics/vendor"]
                if not any(root.resolve() in candidate.parents for root in allowed):
                    raise ValueError("Lab asset not allowed")
                self.file(candidate)
            elif parts[0] == "vendor":
                relative = "/".join(parts[1:])
                allowed = ("pdfjs-dist/build/", "pdfjs-dist/cmaps/", "pdfjs-dist/standard_fonts/", "pdfjs-dist/wasm/", "pdfjs-dist/web/", "katex/dist/", "marked/lib/", "dompurify/dist/")
                if not relative.startswith(allowed) or ".." in parts:
                    raise ValueError("Asset not allowed")
                self.file(contained(HERE / "node_modules", HERE / "node_modules" / relative))
            elif path in ("/", "/index.html", "/app.js", "/reader.js", "/style.css"):
                self.file(HERE / "web" / ("index.html" if path == "/" else path[1:]))
            elif path == "/favicon.ico":
                self.respond(b"", "image/x-icon", 204)
            else:
                self.respond({"error": "Not found"}, status=404)
        except (KeyError, FileNotFoundError, StopIteration):
            self.respond({"error": "Document or asset unavailable"}, status=404)
        except ValueError:
            self.respond({"error": "Invalid request"}, status=400)
        except OSError:
            self.respond({"error": "Could not read local data"}, status=500)

    def do_POST(self):
        if not self.trusted() or self.headers.get("X-Reading-Token") != self.server.token:
            self.respond({"error": "Local session token required"}, status=403)
            return
        if self.path != "/api/progress":
            self.respond({"error": "Not found"}, status=404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 150000 or self.headers.get_content_type() != "application/json":
                raise ValueError("Invalid request size or content type")
            payload = json.loads(self.rfile.read(size))
            self.respond(self.server.store.update(payload))
        except Conflict as exc:
            self.respond({"error": str(exc)}, status=409)
        except (ValueError, KeyError, TypeError):
            self.respond({"error": "Invalid progress record"}, status=400)
        except OSError:
            self.respond({"error": "Save failed. Your notes are still in this tab; retry or copy them before closing."}, status=500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--exports", type=Path, default=WORKSPACE / "Data/Exports/ForPropulsion")
    parser.add_argument("--state-dir", type=Path, default=WORKSPACE / "Data/ReadingRoom/propulsion")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    # Runtime state must never overwrite a source bundle or become repository content.
    target = args.state_dir.resolve()
    for protected in (args.exports.resolve(), REPO.resolve()):
        if target == protected or protected in target.parents:
            parser.error("--state-dir must be outside the repository and exported bundles")
    if not (HERE / "node_modules/pdfjs-dist/build/pdf.mjs").is_file():
        parser.error("Browser assets missing. Run npm ci --ignore-scripts --omit=optional in ReadingRoom first.")
    server = ReadingServer(args.port, args.exports, args.state_dir)
    print("Propulsion reading room: http://127.0.0.1:%d" % server.server_address[1], flush=True)
    print("Progress: " + str(server.store.path), flush=True)
    if args.open_browser:
        webbrowser.open("http://127.0.0.1:%d" % server.server_address[1])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
