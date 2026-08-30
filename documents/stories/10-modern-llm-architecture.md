As an engineer who intends to fine-tune rather than invent architectures, I want to understand the modern language-model stack by rebuilding its pieces, so that when a fine-tune behaves strangely I can tell whether the cause is the data, the objective, or the architecture.

Each architectural component studied MUST be reimplemented from scratch and MUST be checked against a reference implementation on identical inputs, agreeing to a stated numerical tolerance.

The study MUST cover what this repository does not already have: mixture-of-experts routing including the load-balancing loss, grouped-query and multi-query attention, rotary position embeddings, and the placement and choice of normalisation in current models.

The study MUST NOT redo scaled dot-product attention or its tiled and tensor-core kernels. Those are already built and benchmarked in CuLLM, from a scalar implementation through WMMA to CuTe, and rebuilding them would be motion rather than progress.

Every component MUST record its parameter count, its arithmetic intensity, and its memory traffic, so that a later fine-tune can be reasoned about under a 12 GB budget rather than discovered to be infeasible during a run.

A component MUST NOT be recorded as understood on the basis of a passing shape test alone; the check MUST compare values.

Never treat a diagram or a blog post as the specification when the reference implementation is available to read.

For example: a mixture-of-experts layer MUST reproduce a reference router's expert assignments on a fixed batch and a fixed seed, and MUST show its auxiliary load-balancing loss falling as the routing becomes balanced. A grouped-query attention block MUST match full multi-head attention to tolerance when the group count equals the head count, which is the degenerate case that proves the implementation.
