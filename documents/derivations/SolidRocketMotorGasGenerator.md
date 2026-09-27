# Solid rocket motor and solid-propellant gas generator with N pintle valves, from first principles

**Serves:** internal ballistics of a solid rocket motor (thrust curve,
chamber pressure, burn time), and the same grain used as a *gas generator*
whose products fill a plenum that is throttled out through a variable number
N of pintle valves. The pintle-valve system is a solid-propellant divert or attitude-control
system (Sutton Fig. 12–27), a hot-gas pressurization system (Huzel & Huang
Figs. 5-9, 5-10), or a throttleable solid motor (Sutton p. 328).
**Plan:** Phase 8 of `research/reading-program/SIMULATION-READING-PLAN.md`,
`propulsion::solid_ballistics` (Sutton roadmap rank 14).
**Builds on:** `notes/topics/solid-ballistics.tex` (lumped ODE, p_eq, the
n < 1 stability proof, τ). This note does not repeat those proofs. It adds the
gas-generator/plenum/valve network, the temperature laws, erosive burning in
lumped form, and grain geometry.
**Modules:** C++ `Cosmos/Source/Propulsion/SolidRocketMotor/`, Rust
`Cosmos/Rust/cosmos_propulsion/src/solid_rocket_motor/`, golden vectors in
`Cosmos/Rust/golden/solid_rocket_motor_*.tsv`.

Page numbers are printed folios (see `PROPULSION-CORPUS.md` §3 for PDF
offsets). Sutton = 9e, H&P = Hill & Peterson 2e, Turns = 3e, Williams = 2e.

## 0. What each of the six books contributes

| Book | What this model takes from it |
|---|---|
| **Sutton** | Burning law and mass balance Eqs. 12-1…12-7 (pp. 444–447); temperature sensitivity Eqs. 12-8…12-14 (pp. 449–451); erosive burning, Lenoir–Robillard Eq. 12-17 (p. 454); choked mass flow Eq. 3-24 and the area–pressure relation Eq. 3-25 (p. 59); thrust coefficient Eq. 3-30 (p. 62), c* Eq. 3-32 (p. 63); hot-gas-valve solid ACS, Fig. 12-27 (p. 483): *"The chamber pressure rises when any valve is closed."*; pintle throat throttling (pp. 40, 328). |
| **Hill & Peterson** | Choked mass flux Eq. (3.14) (p. 71); the lumped chamber equation with the (ρ_p − ρ_0) filling term, Eq. (12.28) (p. 599); why T₀ is independent of pressure (p. 599). |
| **Humble** | Unsteady lumped-parameter method, Eqs. (6.33)–(6.36) (p. 337), the same ODE written as (1/p) dp/dt; §6.6 case study (p. 352) is the planned end-to-end golden vector (not yet encoded). |
| **Huzel & Huang** | Solid-propellant gas generators: the start cartridge (p. 116: 1000 psia, 4.7 lb/s, 2550 °F, c* = 4260 ft/s, ≈ 1.0 s) used here as the gas-generator test case; solid-GG pressurization with an outlet orifice or hot-gas regulator and overboard dump (pp. 148–149). |
| **Turns** | Perfect-gas constant R = R_u/MW, Eq. (2.3) (p. 13); the open control-volume mass and energy balances of the well-stirred reactor, Eqs. (6.28), (6.34) (pp. 194–196). With no reaction, the plenum is exactly this reactor. |
| **Williams** | The theory behind the burning law. Deflagration of a homogeneous solid, §7.1 (p. 230), and gas-phase-controlled deflagration, §7.5 (p. 243). Eq. (7-41) (p. 250): m = c pⁿ/(T_e − T₀) with the pressure exponent n < 1 "if the propellant is to burn in a stable manner in a rocket chamber". That equation is implemented here as the second temperature law. |

## 1. Combustion products (Turns Ch. 2, Sutton Ch. 5, H&P §2.4)

Chemistry enters through the equilibrium products. The two books' reasons
are in `solid-ballistics.tex` §"How much chemistry". The products are a
calorically perfect gas with stagnation temperature T₀, molar mass 𝓜 and
ratio of specific heats γ:

- R = R_u/𝓜 (Turns 2.3); c_p = γR/(γ−1), c_v = R/(γ−1).
- Γ(γ) = √γ · (2/(γ+1))^{(γ+1)/(2(γ−1))} (H&P 3.14).
- c* = √(R T₀)/Γ (Sutton 3-32).

Some sources give c* and T₀ instead of 𝓜, as Huzel's cartridge does. Invert
the last line: 𝓜 = R_u T₀/(Γ c*)². `CombustionProducts::from_characteristic_velocity`
does this. **Assumption P1:** the gas is the same everywhere in the network,
and T₀ does not depend on pressure (H&P p. 599).

## 2. Burning rate

**Saint-Robert/Vieille law** (Sutton 12-5, H&P 12.25): r = a(T_b) pⁿ.

**Temperature law, Sutton form.** Sutton 12-12 defines σ_p = d ln a/dT_b.
With σ_p constant, integrate from a reference grain temperature T_ref:

  a(T_b) = a_ref · exp(σ_p (T_b − T_ref)).

**Temperature law, Williams form.** Williams (7-41) writes the mass
burning rate m = ρ_b r as c pⁿ/(T_e − T₀). Here T₀ is the initial solid
temperature (our T_b) and T_e is the "explosion temperature". Normalize at
T_ref:

  a(T_b) = a_ref · (T_e − T_ref)/(T_e − T_b),

so σ_p = d ln a/dT_b = 1/(T_e − T_b). This is no longer constant: the
propellant gets more temperature-sensitive as it warms toward T_e. The
two laws agree to first order at T_ref when σ_p = 1/(T_e − T_ref). That is a
test.

**Pressure sensitivity** follows as π_K = σ_p/(1−n) (Sutton 12-14). It is a
test of the equilibrium solution, not an input.

**Erosive burning, lumped (Sutton 12-17, Lenoir–Robillard):**

  r = a pⁿ + α G^{0.8} D^{-0.2} exp(−β r ρ_b/G).

G is the port mass flux and D = 4A_p/S the port hydraulic diameter. The
lumped model needs one G. We take the **aft end of the port**. There,
every kilogram generated upstream passes, so G = ρ_b A_b r/A_p. This is the
largest G in the port, so the lumped erosive rate is an **upper bound** on
the port-averaged augmentation. The substitution simplifies the equation:
β r ρ_b/G = β A_p/A_b does not depend on r. So

  r = r₀ + C r^{0.8},  C = α (ρ_b A_b/A_p)^{0.8} D^{-0.2} exp(−β A_p/A_b),  r₀ = a pⁿ.

**Claim:** for r₀ > 0 and C ≥ 0 this has exactly one root r > 0, and it
satisfies r ≥ r₀. *Proof.* Let g(r) = r − C r^{0.8} − r₀. Then
g(r₀) = −C r₀^{0.8} ≤ 0. Also g(r) → +∞, because r^{0.8} grows slower than r.
There is a root at or after r₀. Uniqueness: g'(r) = 1 − 0.8 C r^{−0.2}
is increasing, so g is convex. g(0) = −r₀ < 0. A convex function that is
negative at 0 crosses zero from below at most once on (0, ∞). ∎
The code brackets [r₀, r₀ + C·max(r₀, r_hi)^{0.8}] and bisects to a stated
relative tolerance. Bisection is used instead of Newton: it is branch-free
at the bracket ends and it produces identical iterates in C++ and Rust,
which keeps the golden vectors tight. β is Sutton's "about 53" (SI) and α is
a named parameter. Sutton 12-18 derives α from heat transfer, but the
inputs are propellant-specific, so α is injected.

## 3. Grain geometry (Sutton §12.3, p. 462; Humble §6.5)

The state is the burned web y ∈ [0, w]. The geometry is the pair of
functions A_b(y), V(y) with dV/dy = A_b (solid-ballistics.tex, eq. sb-volume-rate).
To make that hold by construction, the code computes V as the case interior
minus the propellant volume:

  V(y) = V_case + V_free − V_prop(y),  so dV/dy = −dV_prop/dy = A_b.

That identity is a property test for each grain.

- **Tubular (internal-burning cylinder):** inner radius a, outer radius b,
  length L. Port radius ρ(y) = a + y. With **inhibited ends**:
  A_b = 2πρL, V_prop = π(b² − ρ²)L, w = b − a. With **burning ends**,
  L(y) = L − 2y, and A_b = 2πρL(y) + 2π(b² − ρ²), V_prop = π(b² − ρ²)L(y),
  w = min(b − a, L/2). Differentiating V_prop gives A_b in both cases; the test
  checks it by finite difference. Port area A_p = πρ², D = 2ρ.
- **End-burning (cigarette) grain:** radius b, length L. A_b = πb² (neutral),
  V_prop = πb²(L − y), w = L. This is the usual shape for a gas generator
  (long, steady, low flow).

**Burnout:** for y ≥ w, A_b = 0. There are no slivers. The chamber then blows
down through the nozzle, and the same ODE gives the tail-off.

## 4. Flow through a restriction (Sutton 3-24, 3-25; H&P 3.14)

Isentropic flow from an upstream stagnation state (p_u, T_u) to a downstream
static pressure p_d, through a restriction of geometric area A and discharge
coefficient C_d:

- Critical ratio π* = (2/(γ+1))^{γ/(γ−1)} (Sutton 3-20).
- If p_d/p_u ≤ π*, the flow is **choked**: ṁ = C_d A p_u Γ/√(R T_u) (Sutton 3-24 = H&P 3.14).
- Otherwise it is **subsonic**, with the throat at p_d:
  ṁ = C_d A p_u √( (2γ/((γ−1) R T_u)) [ (p_d/p_u)^{2/γ} − (p_d/p_u)^{(γ+1)/γ} ] ).
  This is Sutton 3-25 solved for the flux at station y. It reaches the
  choked value at p_d/p_u = π* (a continuity test) and zero at p_d = p_u.
- If p_d > p_u, the flow **reverses**: swap the roles, use the downstream
  temperature, and negate the sign.

## 5. Thrust of a nozzle or valve (Sutton 3-25, 3-29, 3-30)

For a choked restriction with exit area ratio ε = A_e/A_t, the design exit
pressure p_e/p_c solves Sutton 3-25 on the **supersonic** branch. The code
solves the area–Mach relation for M_e > 1 by bisection; the equation is
monotone on that branch. Then

  C_F = √( (2γ²/(γ−1)) (2/(γ+1))^{(γ+1)/(γ−1)} [1 − (p_e/p_c)^{(γ−1)/γ}] ) + (p_e − p_a)/p_c · ε   (Sutton 3-30),

and F = C_F p_c A_t C_d. The C_d on flow and thrust is the ideal-nozzle
correction; Sutton's thrust correction factor is 1 here. **Scope:**
over-expanded separation (Sutton §3.3) is not modeled. The formula is the
ideal attached-flow C_F, so a heavily over-expanded valve reads low. If the
restriction is **not choked**, the jet leaves at p_a subsonically and
F = ṁ v_e with v_e from Sutton 3-16 at p₂ = p_a.

## 6. Pintle valve (Sutton p. 328; Huzel valves ch.)

A conical pintle of half-angle θ sits in a throat of radius R_t. At stroke s
(s = 0 is seated), the pintle radius in the throat plane is
r_p = max(0, R_t − s tan θ). The annular flow area is

  A(s) = π (R_t² − r_p²),  A(0) = 0,  A(s ≥ R_t/tan θ) = πR_t².

The opening command is the normalized stroke x = s/s_full ∈ [0, 1], with
s_full = R_t/tan θ, so A(x) = πR_t² [1 − (1 − x)²]. That is quadratic near
closure, the usual conical-pintle characteristic. Each valve has its own
C_d, R_t, exit area A_e ≥ πR_t², and a piecewise-linear schedule x(t). The
valve count N is a runtime size. A valve with A = 0 passes no mass and makes
no thrust.

## 7. The networks and their ODEs

### 7a. Standalone motor (grain chamber → nozzle → ambient)

**Conservative state.** Integrate the chamber gas mass m_c, not p, and
recover p = m_c R T₀/V(y). The mass balance is then

- dm_c/dt = ρ_b A_b(y) r − ṁ_n(p → p_a)   (Sutton 12-3),
- dy/dt = r, with r = r(p, T_b [, erosive]) for y < w, else 0.

The pressure form is a consequence. Differentiate p = m_c R T₀/V and use
dV/dt = A_b r:
dp/dt = (R T₀/V) [ (ρ_b − p/(R T₀)) A_b r − ṁ_n ]. This is H&P (12.28) and
Humble (6.36), with the (ρ_b − ρ) filling term. Why integrate m_c: every
conservation statement below is then **linear** in the state, and RK4
preserves linear invariants to round-off (§7b). The pressure form's
invariant p V(y)/(RT₀) is nonlinear, and RK4 would only keep it to O(h⁴).

Quadratures: dm_out/dt = ṁ_n, dI/dt = F, dm_burned/dt = ρ_b A_b r.

### 7b. Gas generator → orifice → plenum → N pintle valves → ambient

The grain chamber is the same as in 7a, except that its outlet is an orifice
into the plenum at (p_pl, T_pl), not a nozzle to ambient. The plenum is a
fixed volume V_pl holding gas mass m and internal energy E = m c_v T. It is
Turns' well-stirred reactor with no reaction and with heat loss
Q̇ = h_w (T_pl − T_wall):

- dm/dt = ṁ_12 − Σᵢ ṁᵢ   (Turns 6.28)
- dE/dt = c_p T_up ṁ_12 − c_p T_pl Σᵢ ṁᵢ − h_w (T_pl − T_wall)   (Turns 6.34, unsteady form)

Here T_up = T₀ when ṁ_12 ≥ 0 (gas-generator gas in) and T_pl when it
reverses. p_pl = m R T_pl/V_pl, T_pl = E/(m c_v).

**Assumption P2:** the grain chamber holds T₀ (H&P p. 599) even on reverse
flow. Reverse flow only happens after grain burnout while the plenum is
still full, and then it only slows the blow-down.

Quadratures: m_out = ∫Σᵢṁᵢ, I = ∫Σᵢ Fᵢ, m_burned = ∫ ρ_b A_b r.

**Conservation (a test):** at every time,

  m_burned = [m_c(t) − m_c(0)] + [m(t) − m(0)] + m_out.

Every term is a state component, so the invariant is a fixed linear
functional ℓ of the state with ℓ·f ≡ 0. Each RK4 stage adds h·Σ bᵢ kᵢ, and
ℓ·kᵢ = 0 for every stage. So ℓ·y is unchanged to round-off, whatever the
step size. The test asserts it at 1e-9 relative, even across burnout and
valve closures.

**Energy (a test, adiabatic case h_w = 0):** the same argument, with
E_gg = m_gg c_v T₀ and the enthalpy flows, gives
c_p T₀ m_burned − c_p ∫T_pl Σṁᵢ ≈ ΔE_gg + ΔE_pl while ṁ_12 ≥ 0. The
code checks the steady plenum temperature instead: T_pl → T₀. With no wall
loss and a single gas, mixing gas-generator gas into gas-generator gas cannot
change its temperature.

**Why closing valves raises both pressures (Sutton Fig. 12-27):** in steady
state with everything choked, ṁ_gen = ṁ_12 = Σṁᵢ gives
p_pl = ṁ c*/ΣC_dᵢAᵢ. Closing valves lowers ΣA and raises p_pl. While the
orifice stays choked, the grain chamber does not see this: p_gg = p_eq of
the orifice. Once p_pl/p_gg > π*, the orifice unchokes. Then ṁ_12 falls at
fixed p_gg, so p_gg rises until the grain's production matches the flow
again. With n < 1 the new equilibrium is stable (solid-ballistics.tex,
Prop. sb-stability). This is how a solid gas generator is throttled by
downstream valves, and it is what the scenario tests exercise.

## 8. Integrator

This is classical fourth-order Runge–Kutta at a fixed step, with the tableau
named (story 15). Fixed step is chosen so the C++ and Rust twins take the
same steps and the golden vectors agree to round-off. The step must resolve
the fastest time constant, τ = L*/((1−n)Γ²c*) for the grain chamber
(solid-ballistics.tex eq. sb-tau), and the plenum filling time
V_pl/(ΣC_dA Γ√(RT)). The scenario tests take h ≤ τ_min/50. An adaptive
DOPRI5 driver is available in `Numerical/ODE/RKMethods` if a consumer later
needs one. The golden-vector twins deliberately stay fixed-step.

## 9. What is not modeled (yet)

- Quasi-1-D port flow and axial pressure drop (H&P 12.37; Humble §6.5.2).
  The erosive term uses the aft-end G as an upper bound instead.
- Two-phase (Al₂O₃) nozzle loss (H&P §12.6; Sutton §3.5), throat erosion,
  nozzle separation, and slivers.
- Igniter mass addition. The run starts from a stated initial chamber
  pressure, a "lit" grain.
- Valve actuator dynamics. Positions are prescribed schedules; a
  pressure-regulating controller that closes the loop on p_pl (Huzel
  pp. 148–149 "hot-gas regulator"; Sutton Fig. 11-4) is the next step.
- Humble §6.6 case study as an end-to-end golden vector.
