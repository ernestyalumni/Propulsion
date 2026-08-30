As a researcher reading toward a simulation, I want to ask a question in ordinary language and get an answer grounded in my own parsed corpus, so that I can trust the answer and go straight to the printed page it came from.

Every answer the companion returns MUST cite at least one locator from the corpus index, and every cited locator MUST resolve to an existing artifact and position.

When the corpus holds no passage supporting the question, the companion MUST say that the corpus does not cover it, and MUST NOT answer from model knowledge alone.

The companion MUST NOT paraphrase an equation without also returning that equation in the source form recorded in the parsed artifact.

The companion MUST be able to answer from the corpus alone, with no network access to a publisher or a search engine.

Never fabricate a page number, equation tag, section title, or book title that is not in the index.

For example: asked "what condition drives combustion instability?", the companion must return the Rayleigh criterion together with the source equation as recorded for Lieuwen-UnsteadyCombustorPhysics and a locator into that book's reconciled output, rather than a restatement from memory.
