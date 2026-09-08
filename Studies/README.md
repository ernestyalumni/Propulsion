# Worked mathematical-physics studies

Three independent teaching implementations connect the Reading Room, shared
LaTeX derivations, exact SageMath checks, and numerical evidence:

| Topic | Derivation | Numerical oracle |
| --- | --- | --- |
| `oscillator` | Hamiltonian splitting, Störmer–Verlet, exact modified invariant | Analytic oscillator, second-order convergence, long-run energy bound |
| `rigid-body` | SO(3), inertia, Euler equations, conserved momentum | Axisymmetric exact solution, fourth-order convergence, rotation/energy/inertial-momentum errors |
| `nozzle` | Conservation, stagnation relations, area–Mach branches | Exact rational special case, bracketed roots, independent mass-flux residual |

The sources are [mechanics.py](mechanics.py) and [symbolic.py](symbolic.py).
Topic-to-book locators and related historical notes live in
[resources.json](../ReadingRoom/resources.json). New derivations are under
[documents/notes](../documents/notes/README.md). Existing domain modules are
linked for further work; their correctness is not asserted by these checks.

From the repository root, using an absolute output path outside the repository:

```sh
python3 -B Studies/verify.py --output /absolute/path/to/artifacts/verification.json
python3 -B Studies/verify.py --output /absolute/path/to/artifacts/verification.json --sage-image YOUR_INSTALLED_IMAGE_ID
```

The second command uses your **already installed, prebuilt** SageMath image.
It resolves its immutable image ID, uses `--pull=never`, disables container
network access, and mounts only `Studies/` read-only. Nothing is built from
source. Inspect available images with `docker image ls sagemath/sagemath`.
The tested image ID and actual Sage version are recorded in the evidence.
Python files importing Sage run with `sage -python`, not the host's Python.

The verifier records numerical metrics, symbolic assertions, runtime versions,
the image ID and hashes of the derivations/code. Failures preserve the previous
report; changed inputs make it stale in the Reading Room. A numerical-only run
records symbolic evidence as absent. These reports are execution evidence,
not a tamper-proof certificate or a proof of every claim in a linked manuscript.

Sage's `latex()` expressions are included in the JSON report, so exact computed
forms can be compared directly with the manuscript. LaTeX builds do not execute
Sage or run arbitrary notebook cells. The browser exposes no execution endpoint.

## Optional interactive Sage

Monoclaw already maintains `Deployments/DockerBuilds/Math/SageMath/`.
Its historical Jupyter configuration disables authentication and publishes on
all interfaces. For this project use the local runner below instead:

```sh
python3 ReadingRoom/sage_session.py --image YOUR_INSTALLED_IMAGE_ID
```

It opens a foreground, authenticated Jupyter session on `127.0.0.1:8889`.
Use the token URL printed by Jupyter. Study sources are mounted read-only;
scratch notebooks use a separate directory outside the repository. Ctrl+C
stops the session. CLI verification needs no notebook server. Cadabra remains
an optional tensor specialist for future work, including general relativity.
