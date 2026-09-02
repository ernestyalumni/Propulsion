# TASK-13 — Specialist baselines: TFNO and U-Net

**Goal.** Competently tuned specialist surrogates on the TASK-12 harness so
that any transformer result is compared against the architectures the program is
supposed to beat.
**Depends on.** TASK-12. Needs the 3060.
**Effort.** 1–2 days.

## Steps
1. TFNO from `neuraloperator` (already in the venv from TASK-00). Sizes: 1M and 10M parameters.
2. U-Net (4 levels, GroupNorm, SiLU), 1M and 10M parameters.
3. Fixed budget for every model: the same number of optimizer steps and batch size,
   AdamW, cosine schedule; a small learning-rate sweep {1e-4, 3e-4, 1e-3} on
   validation only; 3 seeds at the chosen rate. Record the tuning effort in hours.
4. Evaluate with TASK-12 metrics: 1-step, 20-step, and 50-step rollouts on the held-out split.

## Outputs
- `Surrogates/fields/models/{tfno.py,unet.py}`, `Surrogates/fields/train.py`, `Surrogates/results/13-specialists.{json,md}`, `Surrogates/results/13-REPORT.md`.

## Done when
- The table has TFNO and U-Net at two sizes, three seeds each, with mean and standard deviation per metric, and the tuning protocol written down.
