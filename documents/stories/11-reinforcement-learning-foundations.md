As an engineer who does not yet know reinforcement learning, I want to learn it by reproducing known results, so that I can tell a working training run from one that only looks like it is training.

Every algorithm studied MUST be implemented and MUST reproduce a published result on a standard benchmark, within a stated tolerance, before it is used on anything of ours.

Every reported learning curve MUST show, on the same axes, a random-policy baseline and a from-scratch baseline of comparable or smaller parameter count.

Every implementation MUST assert that its policy's actions actually reach the environment, and MUST fail loudly when the action applied to the simulator is not the action the policy produced.

Every implementation MUST assert that its update signal is not identically zero, and MUST report the magnitude of the parameter update per step alongside the reward curve.

A run MUST NOT be described as reinforcement learning when no policy gradient, value estimate, or search is involved. Return-conditioned behaviour cloning MUST be named as such.

Never report a result from a run whose seed, hyperparameters, and environment version were not recorded.

For example: the two hackathon repositories that prompted this work fail exactly these checks, and they are the reason the checks exist. In one, the update signal is the mean of z-scored returns, which is identically zero by construction, so parameter updates land near 1e-19 and the policy never changes. In the same repository the policy's action is stored to a field that nothing ever reads, while the simulator steps on uniform random torques. Both faults are invisible in a reward plot and both MUST be caught by an assertion rather than by reading the source months later.
