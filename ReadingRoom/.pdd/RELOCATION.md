# Canonical checkout restored — 2026-09-06

Work now lives in the original Propulsion repository:
`/media/propdev/Expansion/openclaw/.openclaw/workspace/repos/Propulsion`.

Historical PDD logs, run evidence and interrupted intent records were copied
unchanged. Any temporary-checkout path in those records describes that earlier
run, not a current execution target. In particular, last_run.json is historical;
do not resume the interrupted agentic apply against its obsolete project_root.
Use the current ReadingRoom directory for new plans and pdd-local.sh commands.
This relocation does not claim that whole-project PDD synchronization completed.

SHA-256 transfer manifests and verification are retained in the main workspace
under `reviews/repository-consolidation-2026-09-06/`. Books, rendered PDFs and
reading progress remain outside Git in the existing Data directories.
