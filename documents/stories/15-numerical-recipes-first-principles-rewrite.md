As an engineer building a multi-physics simulation library from the classical numerical-methods texts, I want every Numerical Recipes section rewritten from its physics and equations rather than from its shipped code, so that each mathematical object is a named, tested, cited type that a reviewer can check against the printed derivation.

Every rewritten method MUST begin from a written statement of the physical system it serves and the mathematical property the algorithm exploits, recorded in a derivation note under documents/ before the code is written. A method whose note does not name the physics MUST NOT be merged.

Every noun in the mathematics MUST become a named type, every constant MUST become a named, asserted, injected parameter, and every verb MUST become a testable function. A numerical literal MUST NOT appear inside a function body unless the note names it and a test asserts it.

The rewrite MUST be derived from the book's prose and equations. Code under NR_C301/code MUST NOT be opened while writing a module, MUST NOT be transliterated, and MUST NOT be vendored, because it is not freely redistributable and its design is the thing being replaced.

Every module MUST carry property tests that follow from the mathematics: order conditions for a Runge–Kutta tableau, reconstruction of the original matrix from a factorization to machine precision, conservation or bounded drift where the physics guarantees it, and observed convergence at the claimed order under step refinement.

Every module MUST carry a citation sidecar naming the Numerical Recipes section, printed and PDF page, and equation tags from the parsed corpus, plus the substitute reference used where the book is dated. The reading of a section MUST be recorded in documents/research/numerical-recipes-rewrite/READING-LEDGER.md and MUST NOT be counted as read until the note, module, tests, and sidecar exist.

Never accept a method as correct because it reproduces the book's printed numbers; accept it when it satisfies the property that the mathematics guarantees.

For example, stepperdopr5.h in the shipped code declares the Butcher tableau as static const literals inside dy() and sets beta to zero inside Controller::success(), silently disabling Lund stabilization. The rewrite in Cosmos/Source/Numerical/ODE/RKMethods names the tableau as ACoefficients, BCoefficients, CCoefficients and DOPRI5Coefficients, names alpha, beta, safety_factor, min_scale and max_scale as asserted constructor parameters of ComputePIStepSize, and a new module MUST follow that pattern, with a test that the tableau satisfies the order-5 conditions and a note that starts from y' = f(t, y) and the stability region, not from the file.
