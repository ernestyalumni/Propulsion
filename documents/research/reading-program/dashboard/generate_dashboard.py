#!/usr/bin/env python3
"""Build dashboard.html from repository state.

Reads: the four progress.json files, the reading ledgers, the parsed-corpus
directories (to detect OCR phases), `cargo test` on Cosmos/Rust, the gtest
binary if built, and `git log`. Writes dashboard.html next to this script.
Publish it with the Artifact tool to the URL in DASHBOARD-URL.md.
"""
import json, os, re, subprocess, datetime, html, glob

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
BOOKS = [
    os.path.join(REPO, "documents/research/numerical-recipes-rewrite/progress.json"),
    os.path.join(REPO, "documents/research/reading-program/Wie-SpaceVehicleDynamicsControl-2e/progress.json"),
    os.path.join(REPO, "documents/research/reading-program/Sutton-RocketPropulsionElements-9e/progress.json"),
    os.path.join(REPO, "documents/research/reading-program/HillPeterson-MechanicsThermodynamicsPropulsion-2e/progress.json"),
]
LEDGERS = {
    "NumericalRecipes-3e": "documents/research/numerical-recipes-rewrite/READING-LEDGER.md",
    "Wie-SpaceVehicleDynamicsControl-2e": "documents/research/reading-program/Wie-SpaceVehicleDynamicsControl-2e/READING-LEDGER.md",
    "Sutton-RocketPropulsionElements-9e": "documents/research/reading-program/Sutton-RocketPropulsionElements-9e/READING-LEDGER.md",
    "HillPeterson-MechanicsThermodynamicsPropulsion-2e": "documents/research/reading-program/HillPeterson-MechanicsThermodynamicsPropulsion-2e/READING-LEDGER.md",
}

def sh(cmd, cwd=None, timeout=600):
    try:
        return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:
        return ""

def ocr_state(book):
    d, stem = book["corpus_dir"], book["pdf_stem"]
    oc = os.path.join(d, "ocr-compare")
    def exists(p): return os.path.exists(p) and os.path.getsize(p) > 0
    marker = (exists(os.path.join(oc, f"{stem}.marker.md"))
              or bool(glob.glob(os.path.join(oc, "*.marker.md")))
              or exists(os.path.join(oc, "marker", "book.md")))
    nougat = bool(glob.glob(os.path.join(oc, "nougat_out", "*.mmd")))
    reconciled = exists(os.path.join(oc, "reconciled", "equations.json"))
    resolved = exists(os.path.join(oc, "reconciled", "equations_resolved.json"))
    running = False
    for log in ("marker.log", "nougat.log", "run_ocr_large.log",
                os.path.join("marker", "marker_book.log")):
        p = os.path.join(oc, log)
        if os.path.exists(p) and (datetime.datetime.now().timestamp() - os.path.getmtime(p)) < 900:
            running = True
    index = exists(os.path.join(d, "INDEX.md"))
    def st(done, prereq=True):
        if done: return "done"
        if running and prereq: return "running"
        return "todo"
    return {
        "index": "done" if index else "todo",
        "marker": st(marker),
        "nougat": st(nougat, marker),
        "reconcile": st(reconciled, marker and nougat),
        "vision": "done" if resolved else ("partial" if reconciled and os.path.exists(os.path.join(oc, "reconciled", "auto_verdicts.json")) else "todo"),
    }

def ledger_rows(rel):
    p = os.path.join(REPO, rel)
    if not os.path.exists(p): return 0
    rows = [l for l in open(p).read().splitlines() if l.startswith("| ") and not l.startswith("| Section") and not l.startswith("|---")]
    return len(rows)

def rust_modules():
    root = os.path.join(REPO, "Cosmos/Rust/cosmos_numerical/src")
    mods = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.rs"), recursive=True)):
        name = os.path.relpath(path, root)
        if os.path.basename(path) in ("lib.rs", "mod.rs", "field.rs"): continue
        src = open(path).read()
        tests = len(re.findall(r"#\[test\]", src))
        golden = "include_str!" in src
        mods.append({"module": name[:-3].replace("/", "::"), "tests": tests, "golden": golden})
    return mods

def cargo_summary():
    out = sh("cargo test 2>&1", cwd=os.path.join(REPO, "Cosmos/Rust"))
    m = re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed", out)
    passed = sum(int(x[1]) for x in m); failed = sum(int(x[2]) for x in m)
    return {"passed": passed, "failed": failed, "ok": bool(m) and failed == 0 and all(x[0] == "ok" for x in m)}

def gtest_summary():
    b = os.path.join(REPO, "Cosmos/BuildGcc/Check")
    if not os.path.exists(b): return None
    out = sh(f"'{b}' --gtest_list_tests 2>/dev/null")
    tests = sum(1 for l in out.splitlines() if l.startswith("  "))
    run = sh(f"'{b}' 2>&1 | tail -3")
    ok = "PASSED" in run and "FAILED" not in run
    return {"tests": tests, "ok": ok}

def git_info():
    return {
        "commit": sh("git rev-parse --short HEAD", cwd=REPO).strip(),
        "branch": sh("git branch --show-current", cwd=REPO).strip(),
        "when": sh("git log -1 --format=%cd --date=short", cwd=REPO).strip(),
        "subject": sh("git log -1 --format=%s", cwd=REPO).strip(),
        "recent": [l for l in sh("git log -8 --format='%ad  %s' --date=short", cwd=REPO).splitlines() if l],
    }

STATUS_LABEL = {"todo": "not started", "reading": "in progress", "read": "read", "module": "module built", "read-only": "read only"}
STATUS_WEIGHT = {"todo": 0, "reading": 0.5, "read": 0.75, "module": 1, "read-only": None}

def pill(kind, text):
    return f'<span class="pill pill-{kind}">{html.escape(text)}</span>'

def book_section(book):
    ocr = ocr_state(book)
    chapters = book["chapters"]
    scored = [c for c in chapters if STATUS_WEIGHT.get(c["status"]) is not None]
    progress = sum(STATUS_WEIGHT[c["status"]] for c in scored) / max(1, len(scored))
    n_module = sum(c["status"] == "module" for c in chapters)
    n_reading = sum(c["status"] == "reading" for c in chapters)
    n_ledger = ledger_rows(LEDGERS[book["slug"]])
    rows = []
    for c in chapters:
        rows.append(
            f'<tr><td class="rank">{c["rank"]}</td>'
            f'<td class="chap"><b>{html.escape(c["number"])}</b> {html.escape(c["title"])}<span class="pg">p. {c["printed_page"]}</span></td>'
            f'<td class="why">{html.escape(c["why"])}</td>'
            f'<td class="mod">{html.escape(c["module"]) or "—"}</td>'
            f'<td class="lang">{html.escape(c["language"]) or "—"}</td>'
            f'<td>{pill(c["status"], STATUS_LABEL[c["status"]])}</td>'
            f'<td class="note">{html.escape(c["notes"])}</td></tr>')
    ocr_pills = "".join(pill("ocr-" + v, f"{k} · {v}") for k, v in ocr.items())
    return f'''
<section class="book" id="{html.escape(book["slug"])}">
  <div class="bookhead">
    <div>
      <h2>{html.escape(book["title"])}</h2>
      <p class="meta">{html.escape(book["authors"])} · {html.escape(book["publisher"])} · {book["pages"]} PDF pages · {html.escape(book["page_offset"])}</p>
    </div>
    <div class="bookstats">
      <div class="bar" title="{progress:.0%} of rewriteable chapters"><div class="fill" style="width:{progress*100:.1f}%"></div></div>
      <p class="meta"><b>{progress:.0%}</b> of rewriteable chapters · {n_module} with modules · {n_reading} in progress · {n_ledger} ledger rows</p>
      <div class="ocr">{ocr_pills}</div>
    </div>
  </div>
  <div class="tscroll"><table>
    <thead><tr><th>Rank</th><th>Chapter</th><th>The physics</th><th>Module</th><th>Language</th><th>Status</th><th>Notes</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
  <p class="src">Roadmap: <code>{html.escape(book["roadmap"])}</code> · corpus: <code>{html.escape(os.path.basename(book["corpus_dir"]))}/INDEX.md</code></p>
</section>'''

def main():
    books = [json.load(open(p)) for p in BOOKS]
    cargo = cargo_summary(); gtest = gtest_summary(); git = git_info(); mods = rust_modules()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    total_chapters = sum(len(b["chapters"]) for b in books)
    total_modules = sum(c["status"] == "module" for b in books for c in b["chapters"])
    total_reading = sum(c["status"] == "reading" for b in books for c in b["chapters"])
    mod_rows = "".join(f'<tr><td class="mod">{html.escape(m["module"])}</td><td class="num">{m["tests"]}</td><td>{pill("module" if m["golden"] else "read", "golden vectors" if m["golden"] else "property tests")}</td></tr>' for m in mods)
    recent = "".join(f"<li><span class='mono'>{html.escape(l[:10])}</span> {html.escape(l[12:])}</li>" for l in git["recent"])
    sections = "".join(book_section(b) for b in books)
    page = f'''<title>Cosmos Reading Board</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#FCFBF8;--surface:#F3F1EA;--ink:#14171C;--muted:#5F676B;--faint:#8A9195;--rule:#DCD9D0;--accent:#0E5B61;--accent-soft:#DFEBEA;
--ok:#2E7D4F;--ok-soft:#DFF0E5;--warn:#9A6A12;--warn-soft:#F6EAC9;--run:#0E5B61;--run-soft:#DFEBEA;--off:#8A9195;--off-soft:#EDECE6;}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--paper:#0F1215;--surface:#171B20;--ink:#E8E9E6;--muted:#A3ABB0;--faint:#78828A;--rule:#262B31;--accent:#55B6BC;--accent-soft:#12292C;
--ok:#6CCB8E;--ok-soft:#14301F;--warn:#E3B341;--warn-soft:#332A10;--run:#55B6BC;--run-soft:#12292C;--off:#78828A;--off-soft:#1E2328;}}}}
:root[data-theme="dark"]{{--paper:#0F1215;--surface:#171B20;--ink:#E8E9E6;--muted:#A3ABB0;--faint:#78828A;--rule:#262B31;--accent:#55B6BC;--accent-soft:#12292C;
--ok:#6CCB8E;--ok-soft:#14301F;--warn:#E3B341;--warn-soft:#332A10;--run:#55B6BC;--run-soft:#12292C;--off:#78828A;--off-soft:#1E2328;}}
*{{box-sizing:border-box}} body{{background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:14.5px;line-height:1.5;margin:0}}
.wrap{{max-width:1240px;margin:0 auto;padding:36px 24px 80px}}
header{{display:flex;flex-wrap:wrap;gap:24px;align-items:flex-end;justify-content:space-between;border-bottom:1px solid var(--rule);padding-bottom:22px;margin-bottom:28px}}
.eyebrow{{font-size:11px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 8px}}
h1{{font-family:Spectral,Georgia,serif;font-weight:700;font-size:clamp(1.7rem,3.4vw,2.4rem);line-height:1.1;letter-spacing:-.015em;margin:0 0 8px;text-wrap:balance}}
.stand{{color:var(--muted);max-width:60ch;margin:0}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;min-width:320px}}
.tile{{background:var(--surface);border:1px solid var(--rule);border-radius:4px;padding:12px 14px}}
.tile .n{{font-family:"IBM Plex Mono",monospace;font-size:1.55rem;font-weight:500;font-variant-numeric:tabular-nums;line-height:1.1}}
.tile .l{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);margin-top:6px}}
.tile.ok .n{{color:var(--ok)}} .tile.bad .n{{color:var(--warn)}}
h2{{font-family:Spectral,Georgia,serif;font-size:1.35rem;font-weight:600;margin:0 0 4px;letter-spacing:-.01em}}
.book{{margin:0 0 44px}} .bookhead{{display:flex;flex-wrap:wrap;gap:20px;justify-content:space-between;align-items:flex-start;margin-bottom:12px}}
.bookstats{{min-width:300px;max-width:420px;flex:1}}
.meta{{color:var(--muted);font-size:13px;margin:0}}
.bar{{height:8px;background:var(--surface);border:1px solid var(--rule);border-radius:4px;overflow:hidden;margin:6px 0}}
.fill{{height:100%;background:var(--accent)}}
.ocr{{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}}
.pill{{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:11px;padding:2px 8px;border-radius:3px;border:1px solid transparent;white-space:nowrap}}
.pill-module,.pill-ocr-done{{color:var(--ok);background:var(--ok-soft);border-color:color-mix(in srgb,var(--ok) 35%,transparent)}}
.pill-reading,.pill-ocr-partial{{color:var(--warn);background:var(--warn-soft);border-color:color-mix(in srgb,var(--warn) 35%,transparent)}}
.pill-read,.pill-ocr-running{{color:var(--run);background:var(--run-soft);border-color:color-mix(in srgb,var(--run) 35%,transparent)}}
.pill-todo,.pill-read-only,.pill-ocr-todo{{color:var(--off);background:var(--off-soft);border-color:var(--rule)}}
.pill-read-only{{font-style:italic}}
.tscroll{{overflow-x:auto;border:1px solid var(--rule);border-radius:4px;background:var(--surface)}}
table{{border-collapse:collapse;width:100%;min-width:900px;font-size:13.5px}}
th{{font-size:10.5px;letter-spacing:.11em;text-transform:uppercase;color:var(--faint);text-align:left;padding:10px 12px;border-bottom:1px solid var(--rule);white-space:nowrap;font-weight:600}}
td{{padding:9px 12px;border-bottom:1px solid var(--rule);vertical-align:top}} tr:last-child td{{border-bottom:none}}
td.rank{{font-family:"IBM Plex Mono",monospace;color:var(--accent);font-variant-numeric:tabular-nums;width:3.2em}}
td.chap{{min-width:200px;font-weight:500}} td.chap b{{color:var(--accent);font-weight:600;margin-right:4px}}
td.chap .pg{{display:block;font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--faint);font-weight:400}}
td.why{{color:var(--muted);min-width:220px}} td.mod{{font-family:"IBM Plex Mono",monospace;font-size:12.5px}} td.lang,td.note{{color:var(--muted);font-size:12.5px}}
td.num{{font-family:"IBM Plex Mono",monospace;text-align:right;font-variant-numeric:tabular-nums}}
.src{{color:var(--faint);font-size:12px;margin:8px 0 0}} code{{font-family:"IBM Plex Mono",monospace;font-size:.9em}}
.two{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px;margin-bottom:40px}}
ul.log{{list-style:none;padding:0;margin:0}} ul.log li{{padding:5px 0;border-bottom:1px solid var(--rule);font-size:13px}} .mono{{font-family:"IBM Plex Mono",monospace;color:var(--faint);margin-right:8px}}
footer{{border-top:1px solid var(--rule);padding-top:16px;color:var(--faint);font-size:12.5px}}
h3{{font-size:.95rem;font-weight:600;margin:0 0 10px}}
</style>
<div class="wrap">
<header>
  <div>
    <p class="eyebrow">Propulsion repository · reading and rewrite program</p>
    <h1>Cosmos Reading Board</h1>
    <p class="stand">Three books read in relevance order and rewritten from first principles into the Cosmos library: Rust first, C++ where the stack needs it, CUDA where the work is data-parallel. Generated {now} from commit <code>{html.escape(git["commit"])}</code> on <code>{html.escape(git["branch"])}</code>.</p>
  </div>
  <div class="tiles">
    <div class="tile"><div class="n">{total_modules}<span style="font-size:.9rem;color:var(--faint)"> / {total_chapters}</span></div><div class="l">chapters with modules</div></div>
    <div class="tile"><div class="n">{total_reading}</div><div class="l">chapters in progress</div></div>
    <div class="tile {"ok" if cargo["ok"] else "bad"}"><div class="n">{cargo["passed"]}</div><div class="l">Rust tests passing</div></div>
    <div class="tile {"ok" if gtest and gtest["ok"] else "bad"}"><div class="n">{gtest["tests"] if gtest else "—"}</div><div class="l">C++ tests {"passing" if gtest and gtest["ok"] else "(not built)"}</div></div>
  </div>
</header>
{sections}
<div class="two">
  <section><h3>Rust modules in <code>Cosmos/Rust/cosmos_numerical</code></h3>
    <div class="tscroll"><table style="min-width:0"><thead><tr><th>Module</th><th>Tests</th><th>Evidence</th></tr></thead><tbody>{mod_rows}</tbody></table></div></section>
  <section><h3>Recent commits</h3><ul class="log">{recent}</ul></section>
</div>
<footer>How this page updates: an agent runs <code>documents/research/reading-program/dashboard/generate_dashboard.py</code>, which reads the four <code>progress.json</code> files, the reading ledgers, the parsed-corpus directories, <code>cargo test</code>, the gtest binary, and <code>git log</code>, then republishes <code>dashboard.html</code> to this URL. Status vocabulary: not started · in progress · read · module built · read only. OCR phases: index · marker · nougat · reconcile · vision.</footer>
</div>
'''
    open(os.path.join(HERE, "dashboard.html"), "w").write(page)
    print(f"dashboard.html written: {total_modules}/{total_chapters} modules, rust {cargo}, gtest {gtest}")

if __name__ == "__main__":
    main()
