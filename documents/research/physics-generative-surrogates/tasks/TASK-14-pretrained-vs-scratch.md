# TASK-14 — Pretrained vs scratch: GPhyT-S and PDE-Transformer-S

**Goal.** The weight-reuse half of the bet, measured: does initializing from
physics-pretrained weights beat the identical architecture from scratch on a
reacting flow neither has seen, at the same budget?
**Depends on.** TASK-12 (and TASK-13 for the comparison table). Needs the 3060.
**Effort.** 2–3 days.

## Models
- GPhyT-S (9.2M): https://huggingface.co/flwi/Physics-Foundation-Model ; code
  https://github.com/FloWsnr/General-Physics-Transformer (MIT). Input is 4 frames at
  256×128 with several physical channels; a neural differentiator plus Forward Euler.
- PDE-Transformer SC-S (~46M) or MC-S (~33M): https://huggingface.co/thuerey-group/pde-transformer (MIT).
Download only these small checkpoints.

## Protocol (initialization is the only intended difference)
1. Adapt input and output channel layers to our fields. For every parameter tensor
   record: retained unchanged / inflated (copied into a wider tensor, rest zero or
   mean) / replaced / frozen / trained. Write the table to `14-parameter-provenance.md`.
2. Two arms per model: pretrained init, random init. Same data, same optimizer,
   same steps, same batch, same seeds (3). Same learning-rate sweep on validation as TASK-13.
3. Evaluate with TASK-12 metrics at 1, 20, 50 steps on held-out data; also report
   validation VRMSE at 2k, 5k, 10k steps to show convergence speed.

## Preregistered margin
Pretrained beats scratch if held-out VRMSE at the end of budget is ≥ 15% lower on
3/3 seeds and spectra and conserved integrals are not worse. Otherwise: no evidence of transfer.

## Outputs
- `Surrogates/fields/models/{gphyt_adapter.py,pdetransformer_adapter.py}`, `Surrogates/results/14-pretrained-vs-scratch.{json,md}`, `Surrogates/results/14-parameter-provenance.md`, `Surrogates/results/14-REPORT.md`.

## Done when
- Both models have both arms at three seeds; the provenance table accounts for every parameter; the margin verdict is stated per model.
