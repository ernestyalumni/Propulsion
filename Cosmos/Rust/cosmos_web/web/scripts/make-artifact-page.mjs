// dist/index.html is a full document for any static host. A claude.ai
// Artifact wraps its page in its own document skeleton, so dist/artifact.html
// is the same page with the <!doctype>, <html>, <head> and <body> wrappers
// removed (their contents kept, in order).
import { readFileSync, writeFileSync } from "node:fs";

const page = readFileSync(new URL("../dist/index.html", import.meta.url), "utf8");
const inner = page
  .replace(/<!doctype html>\s*/i, "")
  .replace(/<html[^>]*>\s*/i, "")
  .replace(/<\/html>\s*/i, "")
  .replace(/<head>\s*/i, "")
  .replace(/<\/head>\s*/i, "")
  .replace(/<meta charset="utf-8">\s*/i, "")
  .replace(/<meta name="viewport"[^>]*>\s*/i, "")
  .replace(/<body>\s*/i, "")
  .replace(/<\/body>\s*/i, "");
writeFileSync(new URL("../dist/artifact.html", import.meta.url), inner);
console.log("wrote dist/artifact.html");
