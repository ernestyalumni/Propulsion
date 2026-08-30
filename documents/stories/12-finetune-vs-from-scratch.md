As an engineer betting that transfer from a pretrained language model beats a specialist trained from scratch, I want that bet run as a controlled experiment, so that it is settled by measurement rather than by preference.

The experiment MUST compare a fine-tuned open-weights model against both a transformer trained from scratch of comparable or smaller parameter count and a competently tuned neural operator, all three on the same task, the same data budget, and the same evaluation.

The fine-tuned model MUST retain its pretrained token embeddings and language-model head. Discarding them removes the pretrained knowledge that is the subject of the experiment, so a run that discards them MUST NOT be reported as evidence about transfer.

The task MUST be stiff, and its stiffness ratio MUST be stated. A benchmark whose dynamics are smooth and analytically tractable MUST NOT stand in for the engineering case.

The evaluation MUST measure autoregressive rollout stability over the horizon the application needs, not single-step prediction error, and MUST report where the rollout diverges rather than reporting only the mean error over a short window.

Every predicted state MUST carry physical units, and the evaluation MUST report conservation of mass, momentum and energy as separate quantities rather than folding them into a single loss.

The result MUST be reported whichever way it falls, and a negative result MUST be written up with the same care as a positive one.

Never tune the proposed approach against the test set while leaving the baselines at their defaults.

For example: the repository cited as evidence for this bet accidentally ran a version of this comparison and lost it. A frozen Pythia-410M with its embeddings replaced scored 74.9 on its benchmark, while a GPT-2 trained from scratch with 700 K parameters scored 409.0 on the same task, roughly five times better at a fraction of a percent of the parameter count. That is the experiment this story runs properly, on a stiff problem, with the embeddings kept.
