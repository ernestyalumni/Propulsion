# TASK-15 — Natural-video arm (Cosmos-1.0 tokenizer) and Walrus memory measurement

**Goal.** Two measurements: (a) whether a natural-video-pretrained tokenizer,
channel-adapted, reconstructs physics fields better than its scratch twin at
fixed compression and budget; (b) whether Walrus (1.3B) can be LoRA-fine-tuned on 12 GB.
**Depends on.** TASK-12. Needs the 3060.
**Effort.** 2 days.

## (a) Tokenizer arm — story 13, narrowed
1. Check whether `nvidia/Cosmos-1.0-Tokenizer-CV8x8x8` (continuous, ~few hundred MB)
   is still downloadable from Hugging Face; record the revision. If it is gone,
   write BLOCKED-15a.md and do part (b) only.
2. Inflate the RGB input/output convolutions to our channel count (record provenance
   as in TASK-14). Scratch twin: same architecture, random init.
3. Train both for the same steps on reconstruction of TASK-12 fields; report per-field
   reconstruction VRMSE in physical units, spectra, compression ratio, GPU memory used.
4. Compare with Sotoudeh et al.'s numbers qualitatively (their −64% in-domain, −19% cross-domain).

## (b) Walrus memory
1. `polymathic-ai/walrus` (1.3B, MIT): load in BF16, run one forward on a TASK-12
   batch, record peak memory; then attach LoRA (rank 8, attention projections)
   with gradient checkpointing, run one training step at batch 1, record peak memory
   and whether it fits in 12 GB. Do not train further.

## Outputs
- `Surrogates/fields/tokenizer_arm.py`, `Surrogates/fields/walrus_memory.py`, `Surrogates/results/15-video-arm.{json,md}`, `Surrogates/results/15-walrus-memory.md`, `Surrogates/results/15-REPORT.md`.

## Done when
- (a) has both arms at fixed compression with the provenance table, or a BLOCKED note; (b) reports peak memory for forward and one LoRA step.
