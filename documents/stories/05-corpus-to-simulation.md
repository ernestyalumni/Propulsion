As a researcher, when the corpus gives me a governing equation, I want to turn it into a runnable simulation in Propulsion that carries its citation, so that the code traces back to the printed source and a reviewer can check it.

Every simulation module derived from the corpus MUST record, in a machine-readable sidecar stored next to the code, the locators it was derived from: book slug, artifact file, and equation tag or page range.

Each recorded locator MUST resolve against the corpus index. A module whose citations no longer resolve MUST fail its own check.

The generator MUST NOT invent a numerical parameter value that does not appear in the cited source. An unknown parameter MUST be surfaced as a required input to the module rather than filled with a default.

Generated simulation code MUST be written inside the repository, and simulation output data MUST be written under the corpus root. Never write simulation output inside the repository working tree.

Never overwrite a hand-written module that has no citation sidecar.

For example: documents/CombustionInstability.md states the Rayleigh criterion and CombustionInstability/Python/combustion_instability_jax.py implements against it, yet neither cites Lieuwen-UnsteadyCombustorPhysics or Natanzon-CombustionInstability, both of which are already parsed. After this story that module carries a sidecar naming the equation tags it came from, and its check fails if those tags stop resolving.
