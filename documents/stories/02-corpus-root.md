As a researcher running Propulsion tooling, when a tool reads a source PDF or writes a parsed artifact, I want the corpus location to come from configuration, so that the repository stays pure code and the data lives on whichever drive currently holds it.

Propulsion MUST resolve the corpus root from a single configured value, read from the PROPULSION_CORPUS_ROOT environment variable when it is set and otherwise from a checked-in configuration file at the repository root.

Every tool that reads or writes corpus data MUST derive its paths from that resolved corpus root.

The location of the external OCR pipeline MUST also come from configuration, and Propulsion MUST NOT assume that pipeline sits at a fixed path relative to this repository.

A Propulsion tool MUST NOT write a parsed data product — OCR output, reconciled markdown, equation JSON, extracted figures — anywhere inside the repository working tree.

A tool invoked with no corpus root configured and no default present MUST fail with a message naming the missing setting, and MUST NOT silently fall back to a path inside the repository.

A tool invoked with a corpus root that does not exist MUST fail with a message naming the path it tried, and MUST NOT create that directory.

Never hardcode an absolute filesystem path to the corpus in Propulsion source, tests, or documentation; use the configured root or the placeholder <CORPUS_ROOT>.

For example: the parsed artifacts for Sutton-RocketPropulsionElements-9e live at <CORPUS_ROOT>/Public/books/EngineeringPhysics/Sutton-RocketPropulsionElements-9e/ocr-compare/. The repository's own Data/ directory holds only small checked-in fixtures (SuiteSparseMatrixCollection, TurbulentCFDExampleCases) and MUST NOT grow a books/ subtree.
