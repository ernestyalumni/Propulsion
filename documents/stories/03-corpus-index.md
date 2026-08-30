As a researcher with a dozen parsed textbooks, when I want to find where a result is stated, I want one queryable index across the whole corpus, so that I can locate a passage by concept instead of remembering which book it was in.

Propulsion MUST build a single corpus index over every source whose parse is recorded complete, and that index MUST be rebuildable from the parsed artifacts alone.

Every index entry MUST carry a locator identifying its source: the book slug, the artifact file, and a position within that file (page range, section heading, or equation tag).

Every locator MUST resolve to an existing file and position at the time the index is built. The build MUST report any entry whose locator does not resolve, and MUST NOT emit that entry into the index.

Indexing MUST NOT re-run OCR and MUST NOT modify any parsed artifact. The index is a derived product written to its own location under the corpus root.

Never index a source that has no recorded complete parse.

For example: an index built today must cover the seven books that currently hold ocr-compare/ directories — Lieuwen-UnsteadyCombustorPhysics, Natanzon-CombustionInstability, Sidi-SpacecraftDynamicsControl, HorowitzHill-ArtOfElectronics3e, Srednicki-QuantumFieldTheory, Goldstein-ClassicalMechanics-3e, Arnold-MathematicalMethodsClassicalMechanics-2e — and a query for "Rayleigh criterion" must return an entry whose locator names Lieuwen-UnsteadyCombustorPhysics and an equation tag that is present in that book's reconciled/equations_resolved.json.
