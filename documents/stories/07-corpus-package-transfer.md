As a researcher with one GPU machine and several machines without one, when a source has been parsed on the GPU machine, I want to export the parsed products as a package and import them elsewhere, so that every machine and every agent works from the same corpus without a second parse.

Export MUST produce a single archive plus a manifest that names every file it contains, each file's size, and a checksum for each file.

The package MUST carry the parse record for the source it holds, so that the importing machine reads the source as already parsed and the pipeline there reports it complete.

Export MUST NOT include the source PDF. The package carries parsed products only.

Export MUST NOT include artifacts that exist only to support a resolution pass and are regenerable from the source, and the package MUST NOT be treated as incomplete for lacking them.

Import MUST verify every checksum in the manifest before writing anything. A package that fails verification MUST be rejected whole, and MUST NOT leave a partial import behind.

Import MUST write only under the configured corpus root. Never write outside it, and never overwrite an existing parse whose record is already complete — a source already present MUST be reported as such and skipped.

For example: Lieuwen-UnsteadyCombustorPhysics holds 65 MB under ocr-compare, of which 58 MB is reconciled/pages and reconciled/sheets — PNG strips rendered for the vision-resolution pass, regenerable from the PDF. Across the seven parsed books that is 315 MB of 437 MB. Excluding them MUST take the corpus package for those seven books to roughly 122 MB, and the reading companion on the importing machine MUST still answer from it.
