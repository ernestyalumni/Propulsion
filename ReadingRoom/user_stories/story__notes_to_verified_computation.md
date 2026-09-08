As a mathematical physicist studying engineering, I want to move from an indexed
book section to a worked LaTeX derivation, its symbolic/numerical code and its
actual evidence, then return to the book, so I can inspect the whole argument.

The notes MUST state assumptions, conventions, domains and limits of evidence.
Shared chapters MUST feed the three companions and thematic master. Sage MUST
use a prebuilt image; never build it from source or copy supplied NR code.
Missing optional references MUST be visible without breaking the reader.
Stale computation/build evidence MUST NOT be shown as current verification.
Navigation and execution evidence MUST NOT alter personal learning checks.

For example, Wie §6.4 links to the torque-free derivation, Sage identities,
RK4 teaching implementation and measured conservation/convergence errors. Its
rotation error is reported, not described as exact group preservation. Editing
that implementation makes its old evidence stale until the checks are rerun.

Approved meaning: Ernest's 2026-09-06 request and approval of the proposed
structure. Behavioral coverage: tests/test_notes.py and tests/browser.cjs.
Specification links: prompts/notes_python.prompt,
prompts/notes_ui_javascript.prompt, prompts/study_artifacts_python.prompt.
No generated story contract or provider semantic check is claimed.
