# TASK-00 — Extend the Surrogates venv and verify the RTX 3060

**Goal.** One working Python environment for every later task, with PyTorch on
the RTX 3060 verified, and the GPU ordering gotcha resolved in writing.
**Why.** Every GPU task depends on it; the 980 Ti will silently break CUDA 13 builds.
**Depends on.** Nothing.
**Effort.** About 1 hour, mostly download.

## Inputs
- `Surrogates/.venv` already exists (uv, Python 3.13.3, cantera 3.2.0, numpy, scipy).
- Host: driver 580.76, CUDA 13.0 toolkit, GTX 980 Ti (nvidia-smi index 0) and RTX 3060 (index 1).

## Steps
1. `cd Surrogates && source .venv/bin/activate`
2. `uv pip install torch h5py pyyaml tqdm matplotlib neuraloperator` — torch resolves to 2.13.0 with `nvidia-cuda-runtime==13.0.96`; accept it.
3. Run and save the output:
   ```bash
   python - <<'PY' | tee results/00-env.txt
   import torch, cantera, numpy, scipy, h5py
   print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
   for i in range(torch.cuda.device_count()):
       p = torch.cuda.get_device_properties(i)
       print(i, p.name, f"{p.total_memory/2**30:.1f} GiB", "sm", p.major, p.minor)
   print("bf16", torch.cuda.is_bf16_supported())
   print("cantera", cantera.__version__, "numpy", numpy.__version__, "scipy", scipy.__version__, "h5py", h5py.__version__)
   PY
   ```
4. Determine which CUDA index is the RTX 3060. If it is not index 0, write the
   line `export CUDA_VISIBLE_DEVICES=<index>` into `Surrogates/env.sh` and
   have every later task source it. If the 980 Ti raises errors on enumeration,
   exclude it the same way.
5. Smoke test on the 3060: a 4096×4096 BF16 matmul, print TFLOP/s.
6. `uv pip freeze > Surrogates/requirements.lock`

## Outputs
- `Surrogates/results/00-env.txt`, `Surrogates/env.sh` (if needed), `Surrogates/requirements.lock`
- `Surrogates/results/00-REPORT.md`

## Done when
- `torch.cuda.is_available()` is True and the matmul ran on a device whose name contains "3060".
- The report states the CUDA index of the 3060 and whether the 980 Ti is visible.

## Do not
- Do not install into the repo-root `.venv` (Python 3.10, used by Corpus tests).
- Do not use conda or system pip. Do not build PyTorch from source.
