# TASK-09 — DeepFlame 2.0 in Docker and the chemistry-share profile

**Goal.** A reproducible container that runs DeepFlame 2.0 with CVODE chemistry
on a stock methane case, and the measured fraction f of wall time spent in
chemistry, which sets the Amdahl ceiling for any end-to-end speedup.
**Depends on.** TASK-00 (Docker 29.1 with the nvidia runtime is on the host).
**Effort.** 1–2 days; expect build failures.

## Facts (verified 2026-09-01)
- No official image or Dockerfile. Documented deps: OpenFOAM-7, LibCantera 2.6,
  Python 3.8, PyTorch cu118 (LibTorch for the C++ DNN interface). Repo:
  https://github.com/deepmodeling/deepflame-dev (v2.0, 2026-01-28). Docs:
  https://deepflame.deepmodeling.com/
- The host owner does not use conda. Build LibCantera 2.6 from source with scons
  inside the image (Cantera tag v2.6.0). If, after two honest attempts, DeepFlame
  only builds with its conda recipe, stop and write BLOCKED-09.md describing that;
  do not install conda on the host under any circumstances.

## Steps
1. `Surrogates/deepflame/Dockerfile` from `ubuntu:20.04`: OpenFOAM-7 from the
   openfoam.org apt repository, build tools, Cantera 2.6.0 from source, LibTorch
   cu118 (CPU LibTorch is acceptable for this task), then DeepFlame v2.0 built per its docs.
2. `Surrogates/deepflame/run.sh` that mounts a case directory and runs a solver.
3. Run a stock DeepFlame methane tutorial (a 2-D laminar or small 3-D case with
   CVODE chemistry). Record the mechanism it uses, cell count, steps, and wall time.
4. Profile chemistry share f: use OpenFOAM's built-in timing or wrap the
   chemistry call with timers (read the chemistry model source to find it); report
   f and the Amdahl ceiling 1/((1−f) + f/S) for S ∈ {10, 100, 1000}.
5. Locate DeepFlame's DNN inference code and document the tensor contract it
   expects from a `.pt` model (inputs, normalization, outputs, dtype). Write it to
   `Surrogates/deepflame/DNN-INTERFACE.md`. This is what TASK-10 conforms to.

## Outputs
- `Surrogates/deepflame/{Dockerfile,run.sh,DNN-INTERFACE.md,BUILD-LOG.md}`, `Surrogates/results/09-chemistry-share.{json,md}`, `Surrogates/results/09-REPORT.md`.

## Done when
- The tutorial case runs to completion in the container and f is reported with the method used to measure it.
