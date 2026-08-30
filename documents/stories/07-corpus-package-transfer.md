As a researcher with one GPU machine and several machines without one, when a source has been parsed on the GPU machine, I want to export the parsed products as a package and import them elsewhere, so that every machine and every agent works from the same corpus without a second parse.

Export MUST produce a single archive plus a manifest that names every file it contains, its path relative to the corpus root, its size, and a checksum.

The package MUST carry the parse record for every source it holds, so that the importing machine reads that source as already parsed and the pipeline there reports it complete.

Including the original source documents MUST be an explicit choice at export time, and the manifest MUST record whether they are present. Source documents are not regenerable, so a package that carries them is expected to be large.

Source documents MUST be selected by content, not by file extension. The corpus holds .djvu as well as .pdf, and a package MUST carry either.

Export MUST exclude artifacts that exist only to support a resolution pass and are regenerable from the source, and the package MUST NOT be treated as incomplete for lacking them. The principle is that a package carries what cannot be regenerated.

A package MUST NOT include anything from a corpus subtree marked private unless I ask for that subtree by name.

The package file MUST be written outside the repository working tree, and MUST NOT be committed to this repository under any mechanism, including large-file storage.

Import MUST verify every checksum in the manifest before writing anything. A package that fails verification MUST be rejected whole and MUST NOT leave a partial import behind.

Import MUST surface every file under the receiving machine's configured corpus root, reconstructing the discipline and slug layout from the manifest's relative paths. That corpus root is the machine's own data directory and is Never the Data/ directory inside this repository, which holds only small checked-in fixtures.

Never overwrite an existing parse whose record is already complete. A source already present on the receiving machine MUST be reported as such and skipped.

For example: the seven parsed books hold 437 MB under ocr-compare, of which 315 MB is reconciled/pages and reconciled/sheets, so a parsed-products package for all seven comes to about 122 MB. Sidi-SpacecraftDynamicsControl's source document is a .djvu, one of thirteen in the corpus, and the 61 source documents across the corpus total 826 MB — so exporting Sidi with its source MUST work, and MUST be a choice I make rather than a default.
