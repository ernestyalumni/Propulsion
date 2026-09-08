# Reading Room: local PDD inference

The saved `.pdd/local_llm.json` selects `http://127.0.0.1:8080/v1` and the
server alias `qwen38-9b-q5-k-m`. No provider or local-server key is required.
Missing API keys alone do not select localhost: this explicit configuration does.

Use the wrapper from any directory:

```sh
bash /path/to/Propulsion/ReadingRoom/pdd-local.sh --estimate-json generate prompts/artifact_boundary_tests_python.prompt --output /tmp/artifact_boundary_candidate.py
bash /path/to/Propulsion/ReadingRoom/pdd-local.sh generate prompts/artifact_boundary_tests_python.prompt --output /tmp/artifact_boundary_candidate.py
```

`pdd-local.sh` enters ReadingRoom, pins its settings file, removes conflicting
endpoint overrides and common provider keys, enables local-only mode, disables
debug context dumps, and invokes Ernest's fork. On another machine set
`PDD_EXECUTABLE=/absolute/path/to/the/fork/.venv/bin/pdd`. Edit the nonsecret
JSON intentionally when changing server/model; use `/v1/models` to get the
advertised alias. It is not necessarily the launch profile or GGUF filename.
Never use `0.0.0.0` as a client destination.

Ordinary `pdd` commands run inside ReadingRoom also discover the JSON, but the
wrapper additionally prevents unrelated environment overrides and GitHub use.
It does not alter global provider configuration or stop/start your LLM server.
The server currently has one slot; serialize requests.

## Supported path and limits

At fork `e02419065`, prompt-file `generate` and `test --manual` use exclusive
local inference, including nested structured code extraction. Estimates make
no inference request. Endpoint/format/truncation failures do not fall back to
cloud or another provider. GitHub-free tracking is separate: `PDD_LOCAL_ONLY=1`
by itself is NOT a prohibition on non-GitHub remote models.

An HTTP chat server is not a tool-capable coding agent. Agentic issue workflows
and agent-only intent/sync stages explicitly stop in local mode. The Reading
Room's approved implementation remains usable; no complete PDD intent-apply,
architecture generation or whole-project synchronization is claimed.

**Known CLI gap:** `test --manual` can return exit code 0 and a checkmark after
an inference error. Require a newly produced nonempty output, inspect the run's
error/model fields and evidence, and execute independent tests. Do not trust
the exit status alone or accidentally reuse an older output file. The local
adapter itself still refuses partial output. The audit has a local issue for
this; the PDD fork source was not changed.

## Linux validation, 2026-09-06

- Health/model discovery succeeded against Ernest's running CUDA server.
- 115 targeted transport/tracker/GitHub-guard/story tests passed, using local
  synthetic HTTP fixtures. The previously failing local-comment steering test
  now passes. This is not the entire PDD test suite.
- `generate` produced six artifact-boundary regression tests using Qwen and
  completed its nested extraction call. API cost reported $0, excluding local
  hardware/electricity. The first output had one bad fixture and one missed
  equality case: transport success did not imply test correctness.
- Tightened the prompt. Broad regeneration was rejected by PDD's test-churn
  guard (0.75 > 0.40), with original tests preserved. Applied only the two
  reviewed corrections directly: test `(repo, repo)`, and do not pre-create
  the output directory when testing non-creation. The generated test file
  identifies these edits; it is not represented as untouched model output.
- The broad `test --manual` request for notes.py was truncated at 8192 tokens;
  no output was accepted. A bounded follow-up uses an import-only facade to
  test the actual `notes.outside_repo`, not a duplicate implementation.
  That command completed locally, including unfinished-output detection and
  extraction. Its four tests produced two passes and two failures because the
  model invented exception-message regexes explicitly excluded by the prompt.
  This raw candidate remains in the workspace audit, not in the accepted suite.
- All 27 Reading Room backend tests pass, including the six reviewed tests.

The workspace audit `reviews/pdd-local-workflows-2026-09-06/` retains the
bounded manual prompt/facade/raw tests. Its `local:2` records the false-success
CLI bug. Source configuration, output existence and executable checks were
verified; no claim of production-ready unattended code generation is made.

Run the adopted tests without LLM/Docker access:

```sh
cd ReadingRoom
python3 -B -m unittest discover -s tests -p test_artifact_boundary_generated.py -v
```

The original independent HTTP/catalog/browser/math checks remain authoritative.
PDD runtime metadata lives under `.pdd/`; failed run evidence is not a
correctness certificate and generation fingerprints may predate reviewed edits.
Generated PDFs, private book bundles and large historical manuscripts are not
automatically sent to the model by these bounded prompt-file commands.
