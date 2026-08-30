As an engineer betting against physics-specific neural architectures, I want a standing adversarial check on that bet, so that I abandon it on evidence rather than defend it because I committed to it.

Every claim made for or against an approach MUST be recorded with the evidence that supports it and a link to the artifact a reader can open.

The adversary MUST test the strongest available version of the approach it is arguing for, using the reference implementation and the authors' own hyperparameters where those are published. A strawman MUST NOT count as a refutation.

Each bet MUST carry a falsifier written down before the experiment runs, naming the benchmark, the baseline, and the margin that would settle it.

A benchmark MUST NOT be accepted when it was chosen only by the side it favours, and a result MUST NOT be reported without the baselines it is being compared against on the same axes.

A claim whose supporting artifact turns out not to do what it was cited for MUST be retracted in the record rather than quietly dropped.

Never let the adversarial check lapse because the bet is currently winning.

For example: the case that motivated this whole bet is combustion chemical kinetics, where reaction rates span orders of magnitude and PINN training diverges. That case MUST be written down as a reproducible benchmark with its actual stiffness ratio stated, and any claim that a physics-informed architecture handles stiffness MUST be run against it and against a classical implicit integrator measured on accuracy per unit compute.
