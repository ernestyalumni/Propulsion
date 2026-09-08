# Propulsion reading room — accepted 2026-09-06

## Approved extension — notes, worked solutions and computation

Ernest approved the proposed structure and requested implementation on
2026-09-06. Add a searchable notes collection with book/section navigation,
LaTeX source and compiled companions/master, SageMath symbolic code, numerical
implementations, and dated verification evidence. Three initial studies cover
an oscillator, torque-free rigid body and isentropic nozzle. Mathematical
rigor has no engineering-imposed ceiling: assumptions, domains, conventions,
derivations and limits of evidence must be explicit.

Shared topic sources MUST feed the three book companions and thematic master,
in portrait and screen editions. Historical manuscripts MUST remain intact;
available legacy sources MUST NOT be called verified merely because linked.
Sage MUST use an existing prebuilt image and MUST NOT be built from source.
Reading progress MUST remain separate from symbolic/numerical evidence.
Changed inputs MUST invalidate verification/build currency. Missing optional
reference repositories MUST NOT break reading. Only registered assets may be
served; browser requests MUST NOT execute code or compile LaTeX. Generated
artifacts and private book bundles remain outside the repository.

The approved intent event is recorded under `docs/intents/`. PDD apply captured
the request but provider-backed architecture generation did not complete;
automatic approval review rejected the external-provider retry. The bounded
specifications under `prompts/` and implementation were prepared directly in
the coding session. They are not claimed as successfully PDD-generated or
synchronized outputs. Existing first-version acceptance below remains binding.

Ernest requested: “Can you implement your recommendation for modest first version?”
The accepted recommendation is a local dashboard with three book cards, a visual
reading roadmap, a PDF reader, durable bookmarks and notes, and links to existing
labs. It supports reading, discussing, deriving, and eventually implementing the
material together. Finder remains available for the original documents.

## Acceptance

- Open Numerical Recipes 3e, Sutton 9e, and Wie 2e from their extracted bundles.
- Resolve paths on this machine, independent of the exporting machine's paths.
- Resume the actual PDF page, zoom, and within-page scroll position after reopening.
- Search indexed sections and jump to their mapped pages. Label estimated mappings.
- Show the ranked reading lists and a proposed dependency roadmap across books.
- Keep reading/discussion/derivation/implementation checks independent. Navigation
  alone MUST NOT assert completion. Imported chapter status MUST NOT become local
  verified progress. Existing code links indicate availability, not a passing test.
- Save section notes, open questions, next actions, and bookmarks on disk outside
  the repository, with a readable session handoff for another agent.
- Originals and exported snapshots MUST NOT be modified or copied into the repo.
- Bind to loopback. Requests MUST NOT expose arbitrary files, follow symlinks out
  of permitted assets, or let another website mutate progress. No source content,
  notes, or telemetry goes to external services. No browser CDN requests.
- Failed/conflicting saves MUST NOT silently discard changes or claim success.
- Corrupt state MUST NOT be silently replaced with empty progress.
- Render math in parsed text, while retaining the PDF as authority for OCR disputes.
- Link the existing quaternion visualization and relevant source code. New physics
  simulations, built-in AI chat, and annotation editing are outside this version.

## Ownership

This checkout has no `.pddrc`, architecture mapping, or matching prompt for this
new module. These files are conventional source, not claimed PDD-generated output.
The exported charter and stories inform corpus handling and learning semantics;
their references to absent generators and Rust modules are historical context.
No existing simulation module is regenerated or modified.
