# Results ledger

One row per claim. A claim without an openable artifact is a hypothesis and
does not belong here. Negative results are recorded with the same care.

| Claim | Evidence | Artifact | Date | Task |
|---|---|---|---|---|
| H2/air chemistry Jacobian stiffness ratio is 10^5–10^8 during induction and ~10^3 at ignition (T0 1000–1500 K, 1–10 atm, φ 0.5–2) | central-difference Jacobian of S(T,Y), eigenvalues, conserved directions excluded | `Surrogates/results/stiffness_benchmark.json` | 2026-09-01 | pre-00 |
| GRI-3.0 CH4/air stiffness ratio is ~10^12 during induction and ~10^5 at ignition (1400 K, 1–10 atm) | same | same | 2026-09-01 | pre-00 |
| Cold-start CVODE costs ~128 µs/cell for h2o2 and ~1.3 ms/cell for gri30 at Δt 1e-6 s, rtol 1e-6/atol 1e-12, Cantera 3.2 Python, single core | timed `reinitialize()`+`advance()` over 400 states | same | 2026-09-01 | pre-00 |
| Warm-start CVODE is 5–200x cheaper than cold start; warm numbers must not be used as the surrogate's denominator | same run | same | 2026-09-01 | pre-00 |
| GRI-3.0 with nitrogen chemistry removed (inert N2, Ar kept) is 36 species / 219 reactions | Cantera species/reaction filter | `Surrogates/README.md` (code in TASK-01) | 2026-09-01 | pre-00 |
| CH4/O2 at O/F 3.6 by mass is φ = 1.11; ideal-gas adiabatic flame T at 300 bar, 800 K inlet is 3958 K | Cantera `equilibrate('HP')` | this ledger | 2026-09-01 | pre-00 |
| In-domain tokeniser pretraining cuts VRMSE 64%, cross-domain physics 19%, natural video untested | Sotoudeh et al. arXiv:2603.05598 | https://arxiv.org/abs/2603.05598 | 2026-09-01 | audit |
| A stiff implicit substep at 16.9% runtime share with a 5.8x-cheaper MLP reached parity at best; Mahalanobis gate deferred 96.8–99.7% of cells | Thümmler & Kuroda arXiv:2608.23075 | https://arxiv.org/abs/2608.23075 | 2026-09-01 | audit |
| PhysiX has released no checkpoints | GitHub README inspection | https://github.com/arshka/PhysiX | 2026-09-01 | audit |
| DeepFlame 2.0 has no official Docker image; install is OpenFOAM-7 + LibCantera 2.6 + Python 3.8 + PyTorch cu118 | README and v1.5 install docs | https://github.com/deepmodeling/deepflame-dev | 2026-09-01 | audit |
