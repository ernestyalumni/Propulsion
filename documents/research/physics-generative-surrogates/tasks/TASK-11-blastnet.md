# TASK-11 — BLASTNet reacting subset: survey, choose, download

**Goal.** One reacting-flow DNS dataset on local disk, chosen for the E2
ablation, with documented fields, mechanism, grid, size, and license.
**Why.** Every candidate pretrained model was trained on The Well; the ablation
needs data none of them saw, and reacting flow is what this repository is about.
**Depends on.** TASK-00. A Kaggle API token must be placed by Ernest at `~/.kaggle/kaggle.json` (none exists).
**Effort.** 4 hours plus download time.

## Steps
1. Survey https://blastnet.github.io/datasets and each reacting configuration's
   Kaggle page. Build a table: name, reacting?, fuel/oxidizer, mechanism, 2-D or
   3-D, grid, number of snapshots, fields stored (u, T, Y_k, p, ρ?), size, license, citation.
2. Choose by these criteria, in order: reacting; fields include T and at least
   major species; smallest size that still has ≥ 50 snapshots of a transient or
   ≥ 20 independent samples of a statistically steady case; subset ≤ 100 GB.
   Prefer H2-air premixed or the H2/CH4 jet flames.
3. Download with the Kaggle CLI (`uv pip install kaggle`) to
   `/media/propdev/Expansion/openclaw/.openclaw/workspace/Data/Public/datasets/blastnet/<name>/`.
   Write `MANIFEST.md`: URL, size, SHA-256 per file, license (CC BY-NC-SA 4.0 — noncommercial), citation.
4. Load one file and print shapes, dtypes, field names, and units. Save to `Surrogates/results/11-blastnet-peek.txt`.

## Outputs
- `Surrogates/results/11-blastnet-survey.md` (the table), the downloaded subset with MANIFEST, `Surrogates/results/11-REPORT.md`.

## Done when
- The survey table covers every reacting configuration listed, the chosen subset is on disk with hashes, and the peek file shows real shapes.

## Do not
- Do not download more than one subset. Do not commit any data.
