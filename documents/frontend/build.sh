#!/usr/bin/env bash
# Rebuild every saved front-end page into documents/frontend/site/.
#
#   documents/frontend/build.sh
#
# 1. regenerates the reading board from repository state (progress.json files,
#    ledgers, corpus directories, cargo test, gtest, git log);
# 2. wraps each artifact body into a standalone HTML document that opens with
#    file://, no account and no network needed;
# 3. rewrites site/index.html.
#
# The bodies stay the published sources: republish them with the Artifact tool
# to the URLs in pages.json. Do not edit anything under site/ by hand.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS="$(cd "$HERE/.." && pwd)"

echo "==> regenerating the reading board"
python3 "$DOCS/research/reading-program/dashboard/generate_dashboard.py"

echo "==> wrapping standalone pages"
python3 "$HERE/wrap_standalone.py" \
  "$DOCS/research/reading-program/dashboard/dashboard.html" \
  "$HERE/site/cosmos-reading-board.html" --emoji "📚" \
  --note "standalone copy; published at https://claude.ai/code/artifact/4b1c3653-ba81-431c-8ebf-70fb60bb5bfd"
python3 "$HERE/wrap_standalone.py" \
  "$DOCS/research/numerical-recipes-rewrite/reading-guide.html" \
  "$HERE/site/numerical-recipes-guide.html" --emoji "🛰️" \
  --note "standalone copy; published at https://claude.ai/code/artifact/3a63d17a-ec6f-4f40-b6f0-f9c93113cb31"

echo "==> index"
python3 "$HERE/make_index.py"

echo "==> done. open: file://$HERE/site/index.html"
