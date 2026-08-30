As a researcher building propulsion simulations, when I add a textbook or paper to my corpus, I want the Propulsion tooling to parse it once and record what it produced, so that running the pipeline again on a source that is already parsed reuses those artifacts instead of parsing it a second time.

The Propulsion repository MUST contain only code, tools, scaffolding and infrastructure. Parsed data products MUST be written outside the repository, under a configured corpus root.

Propulsion MUST invoke the external OCR pipeline rather than reimplement it, and MUST NOT vendor that pipeline's scripts, model weights or virtual environments into this repository.

Parsing MUST be recorded per stage — text extraction, reconciliation, and conflict resolution — and the record MUST name which stage completed and when.

Completeness MUST be read from that recorded outcome. A source MUST NOT be reported as parsed merely because its output directories exist.

Re-running the pipeline against a source whose recorded parse is complete and whose input is unchanged MUST NOT invoke nougat or marker again, and MUST report the source as already parsed.

Re-running against a source whose recorded parse stopped short of a stage MUST run only the stages that are missing.

Never delete or overwrite an existing parse artifact as a side effect of a re-run.

For example: Lieuwen-UnsteadyCombustorPhysics already holds ocr-compare/nougat_out, ocr-compare/reconciled and a .marker.md file, and its reconciled/equations_resolved.json records 373 agreements and 306 conflicts. Re-running the pipeline on that book must skip every stage and report it as already complete.

For example: Arnold-MathematicalMethodsClassicalMechanics-2e holds the same directories, but its reconciled/equations.json records 0 agreements, 0 conflicts, 1 marker-only equation and a nougat page repeated 24 times, and it has no equations_resolved.json at all. That parse failed. It MUST NOT be reported as complete, and a re-run MUST be allowed.
