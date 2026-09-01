As an engineer evaluating a pretrained generative model as a full-field numerical surrogate, I want a controlled long-rollout benchmark, so that any gain from transfer is measured against specialist surrogates and the classical simulator rather than inferred from video quality.

This experiment MUST start only after story 13's tokenizer gate passes, and it MUST use the exact versioned data representation, splits, units, normalization, compression ratio, and evaluation harness established there. A failed or inconclusive tokenizer gate MUST stop the experiment rather than trigger a larger model run.

The benchmark MUST compare the transferred autoregressive model against the same dynamics architecture trained from scratch and against competently tuned specialist baselines, including an FNO or TFNO and a convolutional baseline, under the same data and evaluation budgets. Baseline hyperparameters MUST receive comparable tuning effort.

Evaluation MUST measure autoregressive rollout over the application horizon and MUST report per-field error in physical units, divergence time, spectral error, conservation error, boundary-condition violations, and performance on held-out parameters or initial conditions. Single-step prediction and perceptual video metrics MUST NOT substitute for these quantities.

Any acceleration claim MUST report inference wall time against the named classical solver at matched acceptable error, together with data-generation and training cost. When the ground-truth solver cannot be rerun, the result MUST be labeled a surrogate-to-surrogate accuracy comparison and MUST NOT be advertised as a simulator speedup.

The model MUST expose an applicability check or uncertainty signal and a documented fallback to the classical solver for out-of-distribution or physically inadmissible rollouts. Cosmos3-Edge or another visual world model MUST NOT be treated as ground truth or as physically accurate merely because its generated frames look plausible.

Never rent a multi-GPU cluster or build custom CUDA training infrastructure before the benchmark contract, baselines, stopping rule, and compute budget have been reviewed.

For example, after a pretrained tokenizer wins the story 13 ablation on `turbulent_radiative_layer_2D`, fine-tune the PhysiX-style Cosmos-1.0 autoregressive backbone and compare it with a from-scratch copy, TFNO, and U-Net over the same held-out trajectories. If no runnable Athena++ comparison is available, report rollout accuracy and inference cost but make no claim of end-to-end solver acceleration.
