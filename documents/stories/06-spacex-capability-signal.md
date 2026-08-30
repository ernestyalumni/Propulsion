As an engineer deciding which physics to build next, when I ask for a capability read on SpaceX, I want the tooling to fetch their public newsroom and careers pages and record the engineering capabilities those pages imply, so that I can sequence work against dated evidence rather than against my impression of what they need.

The harvest MUST run only when I ask for it. Propulsion MUST NOT poll SpaceX pages on a schedule, and MUST NOT fetch them as a side effect of any other command.

Every recorded capability MUST carry the source URL it was read from and the date it was retrieved.

The record MUST store the extracted capability statement, not the source text. A fetched job posting or news article MUST NOT be copied into this repository in full.

Every recorded capability MUST be mapped either to a directory in this repository that already covers it, or recorded as a gap with no owner.

A harvest that fails to fetch, or that returns no listings, MUST fail loudly and MUST NOT write a signal file.

Never overwrite a previous signal file. Each harvest is a new record named by its date, so the signal can be read as a trend.

This harvest MUST be the only part of Propulsion that reaches the network. Every other tool MUST keep working with no network at all.

For example: a posting naming combustion stability maps to CombustionInstability/; one naming trajectory optimization maps to Cosmos/Source/Astrodynamics/; one naming cryogenic fluid management maps to nothing in this repository today and MUST be recorded as a gap rather than silently dropped.
