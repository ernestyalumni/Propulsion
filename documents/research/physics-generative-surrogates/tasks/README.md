# Task board: physics surrogates program

Self-contained briefs for AI agents or harnesses of any capability. Each brief
states its goal, dependencies, inputs, steps, outputs, and a checkable
definition of done. An agent should be able to execute one brief from a cold
start after reading only `../SESSION-2026-09-01-FINDINGS.md` and the brief.

## How to launch a task (paste this preamble, then the brief)

```
You are working in Ernest Yeung's Propulsion repository at
/media/propdev/Expansion/openclaw/.openclaw/workspace/repos/Propulsion
on branch feat/corpus-user-stories. Read, in full, before doing anything:
  1. documents/research/physics-generative-surrogates/SESSION-2026-09-01-FINDINGS.md
  2. the task brief you were given
Rules that override everything else:
  - Python only inside a uv venv. For this program:
      cd Surrogates && source .venv/bin/activate && uv pip install <pkg>
    Never use system pip, pip3, or conda.
  - Never commit to master or main. Do not commit at all unless the brief says so.
  - Never edit documents/stories/*. Never edit another task's outputs.
  - Write outputs only to the paths the brief names. Datasets go under
    /media/propdev/Expansion/openclaw/.openclaw/workspace/Data/Public/datasets/.
  - Never change a classical baseline's tolerances or a baseline model's
    hyperparameters after seeing a surrogate result. Never tune on held-out data.
  - Do not download files larger than 5 GB or any checkpoint larger than 1 B
    parameters unless the brief names it.
  - Use the RTX 3060 only. Verify with torch.cuda.get_device_name(0); the
    GTX 980 Ti is unsupported by CUDA 13 wheels.
  - When done, append one row per claim to
    documents/research/physics-generative-surrogates/RESULTS-LEDGER.md
    (claim | evidence | artifact path | date | task id).
  - If blocked after two distinct attempts, write
    documents/research/physics-generative-surrogates/tasks/BLOCKED-<task-id>.md
    with what you tried, exact error text, and what you need, then stop.
```

## Dependency graph

```
TASK-00 env
  ├─ TASK-01 mechanisms ─ TASK-02 truth ─┬─ TASK-03 CVODE baseline
  │                                       ├─ TASK-04 ISAT-lite
  │                                       ├─ TASK-05 Stiff-PINN
  │                                       └─ TASK-07 flow-map model ─ TASK-08 gate + a posteriori ─┐
  ├─ TASK-06 1-D solver ─────────────────────────────────────────────────────────────────────────┤
  ├─ TASK-09 DeepFlame Docker + chemistry share f ─────────────────────────── TASK-10 end-to-end ─┘
  ├─ TASK-11 BLASTNet ─ TASK-12 field harness ─┬─ TASK-13 TFNO/U-Net
  │                                             ├─ TASK-14 pretrained vs scratch
  │                                             └─ TASK-15 natural-video arm + Walrus memory
  └─ TASK-16 agent-reproduction spec (after TASK-08)
```

## Board

| ID | Title | Experiment | Depends on | Effort | Needs GPU | Status |
|---|---|---|---|---|---|---|
| 00 | Extend the Surrogates venv, verify the 3060 | all | — | 1 h | yes | todo |
| 01 | Prepare and cross-check mechanisms | E1 | 00 | 2–4 h | no | todo |
| 02 | Truth generator, splits, label validation | E1 | 01, decision 1 | 1 day | no | todo |
| 03 | Classical baseline: CVODE cost and stiffness at the frozen envelope | E1 | 02 | 2 h | no | todo |
| 04 | Tabulation incumbent (ISAT-lite) | E1 | 02 | 1 day | no | todo |
| 05 | Stiff-PINN reproduction and H2/O2 attempt | E1 | 02 | 1–2 days | yes | todo |
| 06 | 1-D operator-split reaction–diffusion solver | E1 | 01 | 2 days | no | todo |
| 07 | Flow-map surrogate: transforms, projection, training, a priori metrics | E1 | 02 | 2 days | yes | todo |
| 08 | Gate, fallback, a posteriori rollouts, go/no-go table | E1 | 06, 07 | 2 days | yes | todo |
| 09 | DeepFlame 2.0 Dockerfile and chemistry-share profile | E1 | 00 | 1–2 days | yes | todo |
| 10 | Surrogate inside DeepFlame, end-to-end wall time | E1 | 08, 09 | 2 days | yes | todo |
| 11 | BLASTNet reacting subset: survey, choose, download | E2 | 00, Kaggle token | 4 h + download | no | todo |
| 12 | Model-independent field harness and trivial baselines | E2 | 11 | 2 days | yes | todo |
| 13 | Specialist baselines: TFNO and U-Net | E2 | 12 | 1–2 days | yes | todo |
| 14 | Pretrained vs scratch: GPhyT-S and PDE-Transformer-S | E2 | 12 | 2–3 days | yes | todo |
| 15 | Natural-video arm (Cosmos-1.0 tokenizer) and Walrus LoRA memory | E2 | 12 | 2 days | yes | todo |
| 16 | Agent-reproduction contract for E3 | E3 | 08 | 1 day | no | todo |

Decision 1 (mechanism and oxidizer) is Ernest's. Until he confirms, TASK-02
runs with the defaults in `../CANDIDATE-EXPERIMENTS-2026-09-01.md` §6.1.

## Conventions for outputs

- Code: `Surrogates/<area>/` where area ∈ {`mechanisms`, `chem`, `onedim`,
  `baselines`, `fields`, `deepflame`}.
- Results: `Surrogates/results/<task-id>-<name>.{json,md}`; every JSON carries
  `cantera`/`torch` versions, git commit of the repo, date, and a `seed` field.
- Large data: `/media/propdev/Expansion/openclaw/.openclaw/workspace/Data/Public/datasets/<name>/`
  with a `MANIFEST.md` (source URL, size, SHA-256 of each file, license).
- Every task ends with a short `Surrogates/results/<task-id>-REPORT.md`:
  what was done, numbers, what failed, what the next task needs to know.
