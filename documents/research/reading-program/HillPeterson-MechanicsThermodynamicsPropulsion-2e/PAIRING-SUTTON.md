# Reading Sutton *with* Hill and Peterson

**Why pair them at all.** Sutton and Biblarz is a handbook: it states the
result, gives you the coefficient, the data table, and the design practice of
the industry. Hill and Peterson is a derivation: it starts from a control
volume and the conservation laws and *gets* to the result, showing which
assumptions were spent along the way. For the endeavor in
`../README.md` — turning these methods into a library — that difference is the
whole point. Sutton tells you the number a routine must reproduce; Hill and
Peterson tells you what the routine is allowed to assume, and therefore where
its validity ends.

A worked example of the difference, and the reason this pairing exists:
Sutton §8.5 hands you the Bartz correlation for thrust-chamber heat transfer as
a formula with a constant in front. Hill and Peterson Chapter 4 derives the
boundary layer that Bartz is a correlation *of* — momentum-integral method,
Thwaites' relation, the turbulent profile, Reynolds analogy between skin
friction and heat transfer. Implement Bartz from Sutton alone and you have a
formula you cannot extrapolate or debug. Implement it having read Hill and
Peterson Chapter 4 and you know exactly which term stops being true when the
chamber runs a strong pressure gradient.

Both books are parsed and page-anchored:

- Sutton 9e — `Data/Public/books/EngineeringPhysics/Sutton-RocketPropulsionElements-9e/`,
  PDF page = printed + 24.
- Hill and Peterson 2e — `Data/Public/books/EngineeringPhysics/HillPeterson-MechanicsThermodynamicsPropulsion-2e/`,
  PDF page = printed + 12 (body); front matter carries roman folios equal to
  the PDF page.

Page numbers below are **printed** pages in each book.

---

## 1. The pairing table

Read across a row in one sitting. The right-hand column is the only reason
the row exists: what you get from Hill and Peterson that Sutton does not give.

| # | Sutton 9e | Hill and Peterson 2e | What H&P adds that Sutton does not |
|---|---|---|---|
| 1 | **2** Definitions and Fundamentals (26–44) | **1.2** Fluid Momentum and Reaction Force (4), **1.3** Rockets (8); **10.2** Static Performance (470) | Thrust is *derived* by applying the momentum theorem to a control volume around the engine, so the pressure-area term is a consequence rather than an assertion. This is the correct foundation for a unit-bearing `performance` type. |
| 2 | **3** Nozzle Theory and Thermodynamic Relations (45–98) | **3** Steady One-Dimensional Flow of a Perfect Gas (65–92), all of it; then **11.3** Nozzles (520) | Sutton's isentropic relations arrive ready-made. H&P Ch. 3 derives the whole quasi-1-D family — isentropic, constant-area with heat addition (Rayleigh), constant-area with friction (Fanno), normal shocks — from one set of equations. Ch. 3 is the single best foundation in either book for a `gasdynamics` module, because every relation comes with the assumption that produced it. §11.3 then adds what Sutton compresses: two-dimensional and divergence losses, and nozzle contour design. |
| 3 | **4** Flight Performance (99–153) | **10.3** Vehicle Acceleration (472), **10.4** Chemical Rockets (478), **10.6** Space Missions (495); **Appendix VIII** Optimization of Multistage Rockets (729) | Appendix VIII is the derivation of the staging optimum — the Lagrange-multiplier argument behind the staging result Sutton states. It is short, self-contained, and directly implementable with its own worked case as the test vector. |
| 4 | **5** Chemical Rocket Propellant Performance Analysis (154–188) | **2.4** Equilibrium Combustion Thermodynamics; Chemical Reactions (40); **12.3** Equilibrium Composition (572), **12.4** Nonequilibrium Expansion (578) | §2.4 builds equilibrium from Gibbs free energy and the equilibrium constant rather than handing you a code's output. §12.4 treats frozen-versus-shifting expansion as a *rate* problem with a recombination time scale, which is what tells you when the frozen approximation is legitimate. Sutton reports the two bracketing answers; H&P tells you which one applies. |
| 5 | **8** Thrust Chambers (271–343), especially §8.5 heat transfer | **4** Boundary Layer Mechanics and Heat Transfer (93–137), all of it; then **11.4** Rocket Heat Transfer (541) | The keystone pairing. Ch. 4 gives the boundary-layer equations, the momentum-integral (Thwaites) method, separation criteria, the turbulent boundary layer, and Reynolds analogy — that is, the physics Bartz correlates. §11.4 applies it to a rocket chamber, including regenerative cooling. |
| 6 | **10** Turbopumps and Their Gas Supplies (365–398) | **13** Turbomachinery for Liquid-Propellant Rockets (615–649), and behind it **7** Axial Compressors (275–364), **8** Axial Turbines (367–423), **9** The Centrifugal Compressor (425–465) | The largest asymmetry in the two books. Sutton spends 34 pages on turbopumps; H&P spends roughly 200 pages deriving the turbomachinery those pumps are: Euler's work equation, velocity triangles, degree of reaction, radial equilibrium, diffusion limits and stall, and — in §13.3 — inducers and the cavitation/NPSH problem specific to rocket pumps. If a real turbopump model is wanted, it is built from H&P and *calibrated* against Sutton. |
| 7 | **9** Liquid Propellant Combustion and Its Stability (344–364) | **12.5** Liquid-Propellant Combustion Chambers (581), **12.8** Combustion Instabilities (606) | H&P treats atomization, vaporization, and the characteristic times whose ratio sets combustion efficiency, giving the mechanism behind Sutton's L\* and residence-time rules. Read alongside the already-parsed Lieuwen and Natanzon for the acoustics. |
| 8 | **12** Solid Propellant Rocket Motor Fundamentals (434–490) | **12.6** Solid Propellants (589), **12.7** Solid-Propellant Combustion Chambers (598) | H&P derives the burning-rate law and the chamber-pressure equilibrium (including the stability condition on the pressure exponent) rather than tabulating it. Compact — a supplement to Sutton, not a replacement. |
| 9 | **17** Electric Propulsion (620–669) | **14** Electrical Rocket Propulsion (651–684); **10.5** Electrical Rocket Vehicles (490); **Appendix IX** Electrostatic and Electromagnetic Forces (733) | §14.4 The Plane Diode derives the space-charge limit (Child–Langmuir), which is the actual physical ceiling on electrostatic thruster current density. §10.5 derives optimum specific impulse for a power-limited vehicle — the mission-level result that decides an electric-propulsion trade. |
| 10 | **6** Liquid Propellant Rocket Engine Fundamentals (189–243) | **13.1** Introduction (615) | Weak pairing, listed for completeness. Feed systems and engine cycles are Sutton's territory; H&P touches them only as context for the turbomachinery. Use Sutton here. |

### No counterpart in Sutton

Hill and Peterson Part 2 is air-breathing propulsion, which Sutton does not
cover at all:

- **5** Thermodynamics of Aircraft Jet Engines (141–216) — ramjet, turbojet,
  turbofan, turboprop cycle analysis and engine–aircraft matching.
- **6** Aerothermodynamics of Inlets, Combustors, and Nozzles (217–273) —
  subsonic and supersonic inlets, gas-turbine combustors, supersonic
  combustion.

These earn their place in this program for two reasons and not for their own
sake: Chapter 6's supersonic-inlet material is shock-system design work that
uses Chapter 3 hard, and Chapters 7–9 are the theory that Chapter 13's
turbopumps are built on.

### No counterpart in Hill and Peterson

Sutton alone covers: liquid propellant property data (7), engine systems,
controls and calibration (11), solid-motor components and materials (13–15),
hybrids (16), thrust vector control (18), selection (19), exhaust plumes (20),
and rocket testing (21). Chapter 11 and Chapter 21 in particular have no
analogue here — for start transients, calibration, health monitoring, and test
instrumentation, Sutton is the only one of the two that speaks.

---

## 2. A reading order for the pair

Ordered so that nothing depends on something unread. Each block is a sitting
or two; the ranks referenced are those in the two `ROADMAP.md` files.

1. **Vocabulary.** Sutton §2.1–2.3; H&P §1.2–1.3. Ends with thrust and
   specific impulse derived, not asserted.
2. **Compressible flow.** H&P Ch. 3 entire. This is the prerequisite for
   everything downstream in both books, and it is the first Rust module
   (`gasdynamics::quasi1d`).
3. **The ideal rocket.** Sutton Ch. 3 with H&P §11.2–11.3. Sutton supplies
   the thrust coefficient, `c*`, and area-ratio practice; H&P supplies the
   derivation and the loss terms.
4. **Thermochemistry.** H&P §2.4 → Sutton Ch. 5 → H&P §12.3–12.4. Derivation
   first, then the handbook results, then the rate argument that says which
   result to use.
5. **Boundary layers and heat transfer.** H&P Ch. 4 entire → Sutton §8.5 →
   H&P §11.4. The keystone; do not take Bartz before Ch. 4.
6. **Vehicle performance.** Sutton Ch. 4 with H&P §10.3–10.6 and Appendix VIII.
7. **Turbomachinery.** H&P Ch. 7 → Ch. 8 → Ch. 9 → Ch. 13, then Sutton Ch. 10
   as the rocket-specific calibration. The longest block by far, and the one
   where the pairing pays most.
8. **Combustion and stability.** Sutton Ch. 9 with H&P §12.5, §12.8, then
   Lieuwen and Natanzon.
9. **Electric propulsion.** H&P Ch. 14 and §10.5 with Sutton Ch. 17.
10. **Sutton-only.** Ch. 11 (systems, controls, calibration) and Ch. 21
    (testing), read for the test-infrastructure work rather than for a module.

## 3. Where the two books disagree, and which to trust

Record disagreements in the ledgers as they are found. Two are known going in:

- **Vintage.** H&P 2e is 1992; Sutton 9e is 2017. Where they differ on
  hardware, performance data, or what is state of the art — engine cycles,
  electric propulsion, materials — Sutton is current and H&P is historical.
  H&P's *derivations* do not age; its *data* does.
- **Notation.** H&P numbers equations `(4.7a)` with a period; Sutton numbers
  them `(2-1)` with a hyphen. The parsed corpora preserve each book's own
  convention (see the Sutton note in `../README.md` about normalizing tags
  before reconciling). Cite equations in the ledger with the book's own form
  so a citation is never ambiguous.
