#!/usr/bin/env python3
"""Turn an Artifact body into a standalone HTML document.

Pages published with the Artifact tool are written as *bodies*: no doctype, no
<html>, no <head>. claude.ai supplies those at publish time, which means the
same file opened with file:// is not a complete document and carries no title
or favicon. This wraps one into a real document that any browser opens
offline, with the head the artifact runtime would have supplied.

Usage: wrap_standalone.py BODY.html OUT.html [--emoji 📚] [--title "..."]
"""
import argparse, os, re, sys
from urllib.parse import quote

HEAD_RESET = """  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root{color-scheme:light dark}
    body{margin:0;font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
    img{max-width:100%}
    [hidden]{display:none!important}
  </style>"""

def emoji_favicon(emoji):
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
           f'<text y=".9em" font-size="90">{emoji}</text></svg>')
    return f'  <link rel="icon" href="data:image/svg+xml,{quote(svg)}">'

def wrap(body, title=None, emoji=None, note=None):
    # The body's own <title> and font <link>s belong in the head.
    found = re.search(r"<title>(.*?)</title>", body, re.S)
    title = title or (found.group(1).strip() if found else "Page")
    body = re.sub(r"<title>.*?</title>\s*", "", body, count=1, flags=re.S)
    # Every <link> in an artifact body is head material (fonts, icons). Match
    # them individually rather than by line: the board puts two on one line.
    links = re.findall(r'<link\b[^>]*>', body)
    for link in links:
        body = body.replace(link, "", 1)
    head = [HEAD_RESET, f"  <title>{title}</title>"]
    if emoji:
        head.append(emoji_favicon(emoji))
    head += ["  " + link.strip() for link in links]
    if note:
        head.append(f"  <!-- {note} -->")
    return ('<!doctype html>\n<html lang="en">\n<head>\n'
            + "\n".join(head)
            + "\n</head>\n<body>\n"
            + body.strip()
            + "\n</body>\n</html>\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("body"); p.add_argument("out")
    p.add_argument("--emoji"); p.add_argument("--title"); p.add_argument("--note")
    a = p.parse_args()
    out = wrap(open(a.body, encoding="utf-8").read(), a.title, a.emoji, a.note)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(out)
    print(f"  {os.path.relpath(a.out)}  ({len(out):,} bytes)")

if __name__ == "__main__":
    main()
