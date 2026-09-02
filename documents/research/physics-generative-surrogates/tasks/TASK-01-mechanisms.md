# TASK-01 — Prepare and cross-check the chemical mechanisms

**Goal.** Three Cantera YAML mechanisms in `Surrogates/mechanisms/` with recorded
provenance, plus an ignition-delay cross-check table between them.
**Why.** Every E1 number is relative to a named mechanism; provenance must be
reproducible by a stranger.
**Depends on.** TASK-00.
**Effort.** 2–4 hours.

## Mechanisms
1. **h2o2** — copy Cantera's bundled `h2o2.yaml` (10 species). Smoke-test mechanism.
2. **gri30_cho** — GRI-3.0 with nitrogen chemistry removed, N2 and Ar kept inert.
   Build it exactly like this and save with `sub.write_yaml(...)`:
   ```python
   import cantera as ct
   full = ct.Solution('gri30.yaml')
   keep = [s for s in full.species() if 'N' not in s.composition or s.name == 'N2']
   names = {s.name for s in keep}
   rxns = [r for r in full.reactions() if set(r.reactants) | set(r.products) <= names]
   sub = ct.Solution(thermo='ideal-gas', kinetics='gas', species=keep, reactions=rxns)
   assert (sub.n_species, sub.n_reactions) == (36, 219), (sub.n_species, sub.n_reactions)
   sub.write_yaml('Surrogates/mechanisms/gri30_cho.yaml')
   ```
   Note in the README that third-body efficiencies referring to dropped species are lost.
3. **ffcm1** — FFCM-1 (38 species / 291 reactions). Download the CHEMKIN files
   from https://web.stanford.edu/group/haiwanglab/FFCM1/ , record each file's
   exact name and SHA-256, and convert:
   `python -m cantera.ck2yaml --input=<mech> --thermo=<therm> --transport=<tran> --output=Surrogates/mechanisms/ffcm1.yaml`
   Assert 38 species and 291 reactions after loading. If the Stanford page is
   unreachable, the `.cti` at jiweiqi/CollectionOfMechanisms is a fallback only
   if a `cti2yaml` converter is available; record which route was used.

## Cross-check
For each mechanism that contains CH4 (gri30_cho, ffcm1) and for h2o2 with H2,
compute constant-pressure ignition delay (time of max dT/dt) on this grid,
with pure O2 as oxidizer and with air, Cantera `IdealGasConstPressureReactor`,
rtol 1e-8, atol 1e-15:
- T0 ∈ {1000, 1200, 1500, 2000} K; p ∈ {1, 10, 100, 300} atm; φ ∈ {0.5, 1.0, 1.11, 2.0}.
Write `Surrogates/results/01-ignition-crosscheck.json` and a markdown table
with the ratio ffcm1/gri30_cho per condition.

## Outputs
- `Surrogates/mechanisms/{h2o2,gri30_cho,ffcm1}.yaml` and `Surrogates/mechanisms/README.md`
  (source URLs, file hashes, species/reaction counts, date, Cantera version).
- `Surrogates/results/01-ignition-crosscheck.{json,md}`, `Surrogates/results/01-REPORT.md`.

## Done when
- All three YAML files load in Cantera 3.2 with the asserted counts.
- The cross-check table exists for all 64 CH4 conditions and 32 H2 conditions, with any integration failures listed by condition rather than skipped silently.

## Do not
- Do not edit reaction rate parameters. Do not "fix" a mechanism that fails at 300 atm; record it.
