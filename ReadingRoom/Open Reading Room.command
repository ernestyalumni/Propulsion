#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [[ ! -f node_modules/pdfjs-dist/build/pdf.mjs ]]; then
  print 'Install the local reader assets first, from this directory:'
  print 'npm ci --ignore-scripts --omit=optional --no-audit --no-fund'
  read '?Press Enter to close.'
  exit 1
fi
exec python3 -B server.py --open-browser
