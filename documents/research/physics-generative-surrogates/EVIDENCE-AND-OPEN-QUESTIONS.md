# Evidence ledger, corrections, and open questions

**Checked through:** 2026-09-01  
**Rule:** a model card or abstract establishes what its authors report, not that
the result will transfer to our application.

## A. Corrections that the next session must preserve

### A.1 Cosmos tokenizer identity

`nvidia/Cosmos-0.1-Tokenizer-DV8x16x16` is an official but older checkpoint.
The public PhysiX instructions reference
`nvidia/Cosmos-1.0-Tokenizer-DV8x16x16`. Similar names caused confusion in the
discussion.

### A.2 PhysiX is evidence, not the required implementation

The important PhysiX result is its transfer ablation and its method of adapting
a video representation to physical fields. Exact reproduction may be useful as
a baseline, but the program should evaluate current models rather than freeze
itself to the authors’ 2025 stack.

### A.3 The 18.7 GB number

The parameters of a 4B BF16 transformer occupy about 8 GB. NVIDIA’s 18.7 GB
number describes GPU memory for the complete Cosmos-1.0 video pipeline under the
most extensive reported offloading. It is not evidence that Cosmos uses a
larger number format per parameter.

### A.4 Cosmos3-Edge parameter wording

The current Hugging Face card reports 4B parameters for Cosmos3-Edge. Current
NVIDIA training documentation refers to a compact 2B dense backbone in specific
Edge recipes. Do not casually substitute one count for the other; inspect which
components are loaded and trained.

### A.5 GPhyT headline number

An earlier conversational summary recorded a 7x headline. The current arXiv
abstract says up to 29x against specialized learned architectures. Neither is a
classical-solver speedup. Identify paper version, task, denominator, and metric
before citing a number.

### A.6 BF16 on the RTX 3060

The RTX 3060 is compute capability 8.6 and supports BF16 neural-network compute.
That does not make BF16 sufficient for reference integration, conserved
quantities, small residuals, or final scientific evaluation.

## B. Claim ledger

| Claim | Current assessment | Evidence and limits |
|---|---|---|
| PINN residual training is structurally difficult for stiff chemical kinetics. | Strong technical prior; locally derived, but application-specific stiffness must still be measured. | See `LaTeXandpdfs/SourceTermSurrogate.tex`; compare with stiff-kinetics literature rather than treating the derivation as an empirical result. |
| A supervised chemistry flow-map/source-term surrogate is worth trying. | Supported. | [DFODE-Kit paper](https://doi.org/10.1016/j.cpc.2025.110013), [DFODE-Kit repository](https://github.com/deepflame-ai/DFODE-kit), and [ammonia/natural-gas scale-separation paper](https://arxiv.org/abs/2507.08277). Reported speedups remain regime- and baseline-dependent. |
| Video pretraining can transfer to numerical field representations. | Supported by PhysiX, not yet established for our data. | [PhysiX paper](https://arxiv.org/abs/2506.17774) and [public code](https://github.com/arshka/physix). Inspect the exact ablation and surviving weights before repeating its conclusion. |
| PhysiX provides an official ready-to-run fine-tuned checkpoint. | Not found during the session. | Recheck Hugging Face and the project releases; absence from one search is not proof of non-release. |
| GPhyT is a general physics foundation model trained on 1.8 TB. | Author-reported. | [GPhyT paper](https://arxiv.org/abs/2509.13805). Its headline gains are against learned architectures, and the weights/code/license need a fresh audit. |
| Most ML-vs-numerical PDE claims use adequate baselines. | False as a general prior. | [McGreivy & Hakim](https://arxiv.org/abs/2407.07218) report 60/76 surveyed claims used weak baselines. Apply their audit questions to every candidate. |
| “Large Physics Models” is an existing term. | Yes, but broader than numerical surrogates. | [LPM paper](https://arxiv.org/abs/2501.05382), DOI `10.1140/epjc/s10052-025-14707-8`. |
| The older v0.1 discrete tokenizer is “the” current Cosmos tokenizer. | No. | [v0.1 checkpoint](https://huggingface.co/nvidia/Cosmos-0.1-Tokenizer-DV8x16x16), [Cosmos-1.0 checkpoint](https://huggingface.co/nvidia/Cosmos-1.0-Tokenizer-DV8x16x16), and [current Cosmos3 collection](https://huggingface.co/collections/nvidia/cosmos3). |
| Cosmos3-Edge is a current open-weights adaptation candidate. | Yes. Its value as a numerical surrogate is unproven. | [Model card](https://huggingface.co/nvidia/Cosmos3-Edge), [file tree](https://huggingface.co/nvidia/Cosmos3-Edge/tree/main), [NVIDIA Cosmos repository](https://github.com/NVIDIA/cosmos), and [post-training documentation](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md). |
| Cosmos3-Edge accepts arbitrary numerical fields natively. | No. | Its documented generator interface is RGB image/video plus supported action schemas. Field channels require an explicit adaptation. |
| LoRA alone can turn Cosmos3-Edge into an accurate numerical simulator. | Unsupported. | LoRA can adapt parameters; it does not define units, replace the RGB/VAE interface, enforce conservation, or guarantee rollout stability. |
| Cosmos3-Edge’s diffusion path could be useful. | Plausible hypothesis. | Potential advantage: probabilistic ensembles. Potential loss: multiple denoising steps, VAE error floor, and high latency. Must be benchmarked against deterministic models. |
| A Cosmos3-Edge component can be explored on the local 3060. | Plausible but unmeasured. | The repository exposes a ~1.41 GB VAE and ~979 MB vision encoder. Training memory depends on shapes, optimizer, saved activations, and checkpointing. |
| Full Cosmos3-Edge generator training belongs on the 3060. | No current evidence. | NVIDIA documents multi-GPU recipes. A local adapter experiment is distinct from full post-training. |
| RTX 3060 supports BF16. | Yes at the hardware level. | [NVIDIA compute-capability list](https://developer.nvidia.com/cuda/gpus) places RTX 3060 at 8.6; [PyTorch BF16 capability API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html) should verify the installed runtime. |
| Four billion BF16 parameters themselves require 18.7 GB. | False. | About 8 GB for weights. See the full pipeline table on the [Cosmos-1.0 AR-4B model card](https://huggingface.co/nvidia/Cosmos-1.0-Autoregressive-4B). |
| Custom pure-CUDA training is the right first step. | No. | It adds autograd, optimizer, sharding, checkpoint, and mixed-precision risk before model utility is known. Reference training first; custom inference only after a measured need. |

## C. Current Cosmos candidate matrix

| Candidate | Why consider it | Why not assume it wins | Proposed role |
|---|---|---|---|
| Cosmos-0.1 DV8x16x16 tokenizer | Small historical video representation. | Superseded; not the later PhysiX checkpoint. | Usually skip. |
| Cosmos-1.0 DV8x16x16 tokenizer | Known PhysiX path; discrete AR-friendly codes. | Archived generation; exact reproduction can become a distraction. | Historical transfer baseline if needed. |
| Cosmos-1.0 AR-4B | Published PhysiX-compatible backbone. | Old pipeline, large video-token memory, no need to start here. | Reproduction/control candidate, not first download. |
| Cosmos3-Edge VAE | Current, small relative component; could test representation transfer locally. | Native RGB VAE; a reconstruction result may not survive field-channel adaptation. | First current-weight probe after the data harness. |
| Cosmos3-Edge full generator | Current 4B multimodal model with action-conditioned dynamics and official post-training. | Diffusion latency, VAE error, no native units/conservation, likely cloud compute. | Current high-risk full-field transfer candidate. |
| Cosmos3-Nano/Super | Greater capacity. | Much larger, higher compute, and physics data may not support the scale. | Only after Edge supplies a scaling reason. |
| Small transformer from scratch | Clean architecture baseline. | No broad pretraining. | Mandatory control. |
| FNO/TFNO or current specialist | Strong physics-surrogate baseline. | May carry unsuitable bias for the selected problem. | Mandatory fair competitor, not presumed answer. |
| U-Net/ConvNeXt/persistence | Cheap, difficult-to-game baselines. | Limited generality. | Mandatory sanity controls. |
| Classical solver and practical reduced model | Defines truth and actual incumbent. | May be slower; sometimes ground truth cannot be rerun. | Mandatory when making simulator speed claims. |

## D. Questions the next session should answer before proposing stories

### D.1 Use-case selection

1. What decision or workload requires the surrogate?
2. Is the value repeated evaluation, uncertainty quantification, control, inverse
   design, or interactive speed?
3. What error is acceptable in physical units?
4. What is the required rollout duration and time step?
5. At what fallback rate does acceleration disappear?

### D.2 Data

1. Which local solver already emits usable field trajectories?
2. Are `CombustionInstability/`, `CUDACFD/`, or The Well the best first data
   source?
3. Which variables and conserved quantities must be represented?
4. How are boundary conditions and geometry encoded?
5. Can train/validation/test splits hold out entire physical regimes rather than
   neighboring frames from the same trajectory?

### D.3 Representation

1. Is a three-channel invertible packing scientifically meaningful, or only a
   smoke test?
2. Should the VAE be inflated to `C` channels, wrapped with learned adapters, or
   replaced?
3. Which pretrained weights remain after adaptation?
4. Does continuous latent diffusion or discrete autoregression better match the
   required accuracy and latency?
5. What reconstruction error floor is acceptable before dynamics training?

### D.4 Model and training

1. Does official Cosmos3 Edge LoRA cover the generator path, or only another
   recipe/component?
2. Which layers should be frozen, adapted, or trained?
3. Is the action-conditioning path reusable for physical parameters and boundary
   controls without destroying its pretrained geometry?
4. How many trajectories and seeds are needed for a fair scratch comparison?
5. What is the actual measured GPU memory at the selected resolution and frame
   count?

### D.5 Evaluation

1. Per-field error after denormalization?
2. Conserved integral error?
3. Spectral and phase error?
4. Boundary-condition violations?
5. Divergence time over autoregressive rollout?
6. Held-out parameter and geometry performance?
7. Calibration or applicability detection?
8. Inference wall time versus the classical solver at matched acceptable error?
9. Total cost including truth generation and training?

## E. Standing adversarial rules

- Read the implementation, not only the abstract or model card.
- Name the baseline behind every “x-times” claim.
- Separate one-step accuracy, rollout accuracy, and wall-clock acceleration.
- Never substitute perceptual video metrics for physical-field metrics.
- Report which pretrained parameters survived interface changes.
- Give scratch and specialist baselines comparable tuning effort.
- Preserve negative results.
- Stop when the representation gate fails; do not answer failure by renting a
  larger cluster.
- Do not claim a model learned governing laws merely because it interpolates a
  solver-generated dataset.

## F. Addendum 2026-09-01

See [CANDIDATE-EXPERIMENTS-2026-09-01.md](CANDIDATE-EXPERIMENTS-2026-09-01.md)
for the full pass. Corrections to carry forward:

- **A.5:** GPhyT v4 abstract says "more than 7x"; the metric is next-step NMSE
  and the denominator is DPOT. The "29x" figure was not found in v4.
- **Section C is incomplete.** Physics-pretrained open weights that fit the
  RTX 3060 now exist and dominate Cosmos3-Edge as numerical-field candidates:
  GPhyT (9.2M/112M/385M, MIT), PDE-Transformer (33M–701M, MIT), Walrus (1.3B,
  MIT). Cosmos3-Edge is demoted to a possible natural-video arm only.
- **Story 13 is partly answered** by Sotoudeh et al. (arXiv:2603.05598):
  in-domain tokeniser pretraining −64% VRMSE, cross-domain physics −19%,
  natural video untested. The open question is natural video vs physics
  pretraining, not pretraining vs none.
- **PhysiX checkpoints:** still not released as of 2026-09-01.
- **Measured:** H2/air stiffness ratio 10^5–10^8 during induction, ~10^3 at
  ignition; GRI-3.0 methane 10^12 and ~10^5. Cold-start CVODE per cell:
  ~130 µs (H2, 10 species) and ~1.3 ms (GRI-3.0, 53 species) at rtol 1e-6.
- **New negative result to preregister against:** Thümmler and Kuroda
  (arXiv:2608.23075) — Amdahl ceiling from runtime share, OOD gating economics
  d_break = (1 − g − r)/(1 − r), offline error does not rank surrogates.
