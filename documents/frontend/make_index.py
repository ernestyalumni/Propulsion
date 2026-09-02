#!/usr/bin/env python3
"""Write site/index.html: the offline landing page listing every saved page."""
import datetime, html, json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PAGES = json.load(open(os.path.join(HERE, "pages.json")))

def commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"

cards = []
for p in PAGES:
    size = os.path.getsize(os.path.join(HERE, "site", p["file"])) if os.path.exists(os.path.join(HERE, "site", p["file"])) else 0
    cards.append(f'''    <a class="card" href="{html.escape(p["file"])}">
      <span class="mark" aria-hidden="true">{p["emoji"]}</span>
      <span class="body">
        <span class="t">{html.escape(p["title"])}</span>
        <span class="d">{html.escape(p["description"])}</span>
        <span class="f">{html.escape(p["file"])} · {size//1024} KB · source <code>{html.escape(p["source"])}</code></span>
      </span>
    </a>''')

page = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Propulsion Pages</title>
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%97%82%EF%B8%8F%3C/text%3E%3C/svg%3E">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
  <style>
    :root{{color-scheme:light dark;--paper:#FCFBF8;--surface:#F3F1EA;--ink:#14171C;--muted:#5F676B;--faint:#8A9195;--rule:#DCD9D0;--accent:#0E5B61;}}
    @media (prefers-color-scheme:dark){{:root{{--paper:#0F1215;--surface:#171B20;--ink:#E8E9E6;--muted:#A3ABB0;--faint:#78828A;--rule:#262B31;--accent:#55B6BC;}}}}
    *{{box-sizing:border-box}}
    body{{margin:0;background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.55}}
    .wrap{{max-width:760px;margin:0 auto;padding:48px 24px 72px}}
    .eyebrow{{font-size:11px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 10px}}
    h1{{font-family:Spectral,Georgia,serif;font-weight:700;font-size:2rem;line-height:1.1;letter-spacing:-.015em;margin:0 0 10px}}
    .stand{{color:var(--muted);margin:0 0 32px;max-width:60ch}}
    .cards{{display:flex;flex-direction:column;gap:14px;margin-bottom:36px}}
    .card{{display:flex;gap:16px;align-items:flex-start;padding:18px 20px;border:1px solid var(--rule);border-radius:5px;background:var(--surface);text-decoration:none;color:inherit}}
    .card:hover{{border-color:var(--accent)}}
    .card:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
    .mark{{font-size:1.5rem;line-height:1.2}}
    .body{{display:flex;flex-direction:column;gap:4px;min-width:0}}
    .t{{font-family:Spectral,Georgia,serif;font-size:1.15rem;font-weight:600}}
    .d{{color:var(--muted);font-size:14px}}
    .f{{color:var(--faint);font-size:12px;font-family:"IBM Plex Mono",monospace;word-break:break-all}}
    code{{font-family:"IBM Plex Mono",monospace}}
    footer{{border-top:1px solid var(--rule);padding-top:18px;color:var(--faint);font-size:13px}}
    footer p{{margin:0 0 8px}}
  </style>
</head>
<body>
  <div class="wrap">
    <p class="eyebrow">Propulsion repository · offline copies</p>
    <h1>Propulsion Pages</h1>
    <p class="stand">Every page this project publishes, saved as a standalone file that opens in any browser with no account and no network. Generated {datetime.datetime.now():%Y-%m-%d %H:%M} from commit <code>{commit()}</code>.</p>
    <div class="cards">
{chr(10).join(cards)}
    </div>
    <footer>
      <p>Rebuild after any change: <code>documents/frontend/build.sh</code>. It regenerates the board from repository state, wraps each artifact body into a standalone document, and rewrites this index.</p>
      <p>Web fonts load from Google Fonts when online and fall back to system serif, sans, and monospace faces when offline. Nothing else is fetched.</p>
    </footer>
  </div>
</body>
</html>
'''
open(os.path.join(HERE, "site", "index.html"), "w", encoding="utf-8").write(page)
print(f"  site/index.html  ({len(page):,} bytes, {len(PAGES)} pages)")
