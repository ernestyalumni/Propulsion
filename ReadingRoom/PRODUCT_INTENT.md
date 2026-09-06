# Propulsion reading room — accepted 2026-09-06

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
