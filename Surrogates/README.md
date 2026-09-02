# Surrogates — measurements for the physics-ML bet

Evidence-gathering code for `documents/research/physics-generative-surrogates/`.
Nothing here trains a model.

## Environment

```bash
cd Surrogates
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install cantera numpy scipy
```

Installed 2026-09-01: Cantera 3.2.0, NumPy 2.5.2, SciPy 1.18.1.

## `stiffness_benchmark.py`

Measures what story 09 requires before any surrogate is trained:

- the stiffness ratio of the chemistry source term along constant-pressure
  ignition trajectories (H2/air `h2o2.yaml`, CH4/air `gri30.yaml`), reported
  at its worst point, at ignition, and as a median, with the conserved
  eigen-directions excluded and counted;
- the cost of the tuned classical integrator (Cantera CVODE) per chemistry
  substep, warm-start and cold-start. Cold-start is the denominator an
  operator-split surrogate must beat.

```bash
python stiffness_benchmark.py     # ~2 min single core; writes results/stiffness_benchmark.json
```

Results and interpretation: `documents/research/physics-generative-surrogates/CANDIDATE-EXPERIMENTS-2026-09-01.md` section 2.
