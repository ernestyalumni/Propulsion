# Notes, worked solutions, and the Propulsion master

`master.tex` assembles shared topic files by subject. The three entry points in
`companions/` assemble the same files for Numerical Recipes 3e, Wie 2e and
Sutton 9e. Compile from **this directory**, or use the builder from the repo root:

```sh
python3 -B ReadingRoom/build_notes.py --output /absolute/path/to/artifacts
```

The builder produces all eight PDFs: a portrait US Letter edition and a
1200 × 700 TeX-point, two-column screen edition of each document. It runs
pdfLaTeX/BibTeX with shell escape disabled, fails on unresolved references or
overfull boxes, keeps intermediates in a temporary directory, and publishes
PDFs with input/output hashes only after every document builds successfully.
Choose the same artifacts directory for `Studies/verify.py` and the reader's
`--artifacts-dir`. Build tools: `pdflatex`, `bibtex`, and the packages named in
`preamble.tex` (a conventional TeX Live installation).

The master begins with mathematical foundations, oscillator integration,
SO(3) rigid-body mechanics, a nozzle derivation, and an index of earlier
manuscripts. It is not a claim of complete coverage of the three textbooks.
Historical notes remain in `LaTeXandpdfs/` and external repos. Reviewed material
can be extracted into topic chapters incrementally without moving originals
or duplicating mathematics between companions.

## Adding a topic

Write a shared `topics/<id>.tex` with `\topic{<id>}{Title}`. State spaces,
domains, regularity, conventions, units, initial/boundary data, approximation
regime, derivation, and scope of evidence. Distinguish physical assumptions,
proofs, symbolic identities, finite numerical checks and experimental evidence.
Use the shared bibliography for citations. Add the chapter to its companion(s)
and the master; add registered source/PDF/code assets and resolvable book-section
locators to `ReadingRoom/resources.json`.

For executable studies use [Studies](../../Studies/README.md). New symbolic work
uses SageMath; no Sage source builds or commercial CAS dependency. Record
synthetic example inputs explicitly. Resolve disputed OCR equations against
the original PDF. Do not copy the book corpus or supplied Numerical Recipes
code into this repository. Existing legacy manuscripts are reference material,
not automatically verified derivations.

The reader distinguishes current builds, stale builds, available source,
missing external assets, and numerical/symbolic evidence. Its personal learning
checkboxes remain independent. Generated PDFs and evidence live outside Git;
all content, build/verification programs and the resource registry are source.
