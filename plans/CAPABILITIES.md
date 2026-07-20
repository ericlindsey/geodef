# New Capabilities — Menu and Discussion

This is a menu of things GeoDef cannot do at all today: new physics, new
source types, new inference methods, new problem classes. The companion
menus are `plans/ARCHITECTURE.md` (internal structure) and
`plans/FEATURES.md` (improvements to existing capabilities).

These items carry more uncertainty than the other menus. Several need a
prototype, a literature check, or an explicit user decision before a
committed plan would be honest; those prerequisites are stated per item,
and the standing questions that need *your* decision (not an experiment)
are collected at the end. Ground rules inherited from the roadmap: new
physics rides the engine/operator interfaces rather than adding special
cases to beginner-facing code; everything is an optional extra; and every
new engine must cite its equations, preserve a reference implementation,
cross-validate published cases, declare capabilities and coordinate
conventions, and show one end-to-end example through the same high-level
workflow.

---

## How to choose from this menu

| # | Item | Value in one line | Effort | Needs first |
|---|---|---|---|---|
| C1 | SMC sampling + model comparison | Multimodal posteriors and evidence — beyond NUTS's reach | Medium | A6 |
| C2 | Meade triangular engine | An independent cross-check on the tri engine | Medium | A1; port experiment |
| C3 | Volcanic point/compound sources | A whole new application domain (volcano deformation) | Medium | A1; a design decision |
| C4 | Tensile/opening slip | Dikes, sills, and opening cracks on existing meshes | Medium | audit note |
| C5 | Multi-fault systems | Joint inversion across named segments | Medium | — |
| C6 | Earthquake-cycle modeling | Simulate sequences of earthquakes, not just one | Large | design note; F9; source recovery |
| C7 | Layered half-space engine | Drop the uniform-elasticity assumption | Large | dependency decision |
| C8 | Mesh refinement and transfer | Meshes that adapt to what the data can resolve | Medium | prototype as example |
| C9 | Elastic-parameter sensitivity | Honest uncertainty from imperfect μ, ν | Small–medium | F9 |
| C10 | Kinematic time-dependent slip | Slip histories from multi-epoch data | Large | F5 epoch bridge |

**A recommended reading.** C1 and C2 are the bounded picks: both build
directly on existing strengths (the Bayesian stack; the engine registry),
both have clear validation stories, and C2 doubles as the acceptance test
for making engine registration public. C4 and C5 are demand-driven — do
them when a study needs them. C3, C6, and C7 each need a decision from you
before serious work starts (see the standing questions). C8–C10 are
deliberately later.

---

## C1. Tempered SMC sampling and model comparison

**What.** NUTS (the current sampler) is excellent on smooth unimodal
posteriors and structurally unable to move between separated modes — and
fault geometry posteriors are exactly where multimodality shows up (dip
ambiguity, conjugate planes). Sequential Monte Carlo (SMC) anneals a
population of samples from the prior toward the posterior, handling
multimodality and producing a log-evidence estimate — the quantity model
comparison needs — as a byproduct.

**Plan sketch.** `bayes.sample_smc(post, *, n_particles, seed,
target_ess=0.5, mutation='nuts', ...)` around BlackJAX adaptive tempered
SMC, working on the same posterior objects `sample` accepts. Requires the
sampler-independent result contract (A6) first, and a small internal
refactor so the collapsed posteriors expose a `log_prior`/
`log_likelihood` split for tempering. Validation ladder, in order:
analytic targets (Gaussian with closed-form evidence; a curved target; a
two-component mixture — the case NUTS structurally fails); NUTS parity on
a unimodal `RectPosterior`; an evidence check against the analytic
marginal likelihood ABIC is built on; then a deliberately multimodal
geometry example in `examples/`. Evidence differences are reported with
honest error bars (variance across seeds) and an explicit
prior-sensitivity warning.

**Open question.** `SlipPosterior`'s positivity truncation makes the
prior normalization intractable; likelihood-only tempering with the
truncated prior as base measure avoids needing it — confirm BlackJAX
supports that on a toy problem before committing the API. Settled:
mutation kernel is NUTS, with random-walk fallback only if that proves
fragile.

**Value.** Medium-high for research use: this is the difference between
"we sampled the mode we started near" and "we characterized the
posterior," plus principled model comparison. Medium effort.

## C2. Meade (2007) triangular dislocation engine

**What.** A second, independent implementation of triangular
dislocations, as a pure cross-check on the Nikkhoo & Walter `tri` engine
— same capability surface, same `Fault.from_triangles` geometry,
`engine='meade'`.

**Why.** Independent cross-validation is the strongest verification
argument available for this class of code, valuable to every downstream
result. It is also the first real test of the engine registry (A1): if
adding Meade requires touching more than the registration plus the port
and its adapter, the registry design failed and should be fixed then.

**Plan sketch and caveats.** Port from Meade's published Matlab (check
the license header first). Known artifact cases near element edges
differ from Nikkhoo — the cross-validation must *document* disagreements,
not hide them. **Experiment first:** port a minimal displacement path and
compare against `tri` on the existing reference geometries before
committing to the full port. Medium effort, low design risk.

## C3. Volcanic point and compound sources

**What.** Mogi point source, McTigue finite sphere, and the compound
dislocation model (CDM/pCDM, Nikkhoo et al. 2017) — the standard toolkit
for volcanic deformation, opening a genuinely new user community.

**The real problem is not the kernels** (they are short); it is that
these sources are not `Fault`s: no patches, no slip vector, no
strike/dip — the unknowns are pressure or volume change plus geometry.
Options: (a) generalize `Fault` — rejected on sight, it muddies the core
class; (b) a parallel `Source` domain object with its own small
forward/inverse surface; (c) engine-level functions only, inversion via
the generic operator interfaces. **Prototype (c) first** — it adds no
public classes, and the object-budget principle demands the justification
before (b) is considered. The data side (GNSS/InSAR/Vertical, noise,
nuisance) applies unchanged. Medium effort once the design decision is
made; needs your input (standing question 2).

## C4. Tensile/opening slip components

**What.** A third slip component (opening) for dikes, sills, and tensile
cracks. The blocked `2N` convention (`[:N]` strike, `[N:]` dip) is an
enduring design convention, so opening must be opt-in without disturbing
it: `components='tensile'` (or `'full'`) yields a `3N` system,
`slip.pack`/`unpack` gain an optional third argument, result views add
`opening`, and moment semantics (tensile versus shear moment tensor) are
made explicit.

**Prerequisite.** A mini design note and audit first: every consumer of
the blocked layout (regularization block structure, plotting, bounds
expansion, result schema) is touched, and the audit must confirm the
default `2N` path stays literally unchanged — the tutorials' contract.
Medium effort; demand-driven.

## C5. Multiple faults and segments

**What.** Solve over a list of `Fault`s with named model partitions — the
model-side analogue of dataset names: per-fault slip views, per-fault
moment, one joint `G`. Continuity constraints across segment joins are
expressed through the existing `constraints=(C, d)` machinery, with a
helper that builds continuity rows from segment adjacency.

**Design stance.** The list form — `solve([fault_a, fault_b], ...)` — and
no new `FaultSystem` class, until cross-call invariants (shared frames,
adjacency) prove one is needed. Medium effort, no hard prerequisites;
demand-driven and a natural companion to C4 for rupture studies.

## C6. Quasi-dynamic earthquake-cycle modeling

**What.** The largest item on any menu: an optional `geodef.cycle` module
simulating sequences of earthquakes — rate-and-state friction on the
existing fault meshes, driven by the elastic stress kernels GeoDef
already computes, with quasi-dynamic radiation damping in place of full
inertia. This turns GeoDef from a static-snapshot tool into one that can
ask how faults behave over many cycles.

**Committed first step: a standalone design note** fixing scope, state
variables, sign/unit conventions, validation targets, and the boundary
between static GeoDef and `geodef.cycle`. The note must resolve, with
small experiments where flagged:

1. **Stress-kernel completeness** — does `fault.stress_kernel` provide
   rake-resolved shear traction and (if needed) normal-stress changes?
   Audit against `unicycle`'s kernels; decide whether v1 fixes normal
   stress. *(experiment)*
2. **Friction-law scope** — aging law only for validation; slip law as
   the second registered law once the callable interface exists.
3. **Validation targets** — `unicycle` runs for port equivalence *and*
   one SEAS community benchmark (BP1-QD is the natural first target);
   confirm a BP1-scale run is tractable in NumPy. *(experiment)*
4. **Integrator** — `scipy.integrate.solve_ivp` (LSODA/BDF; the system
   is stiff between events) versus a custom adaptive RK like
   `unicycle`'s; benchmark both on the spring-slider. *(experiment)*
5. **Units and time** — SI seconds internally with year conveniences at
   the interface, or years throughout; must match the coupling APIs'
   m/yr vocabulary. Settle before any code.
6. **Checkpoint format** — whether restart invariants justify a state
   record class; defer until a real restart shows what it needs.
7. **Source recovery** — `related/stress-shadows/unicycle/` is absent
   from fresh clones; confirm attribution and decide where the reference
   sources live (standing question 1).

**Deliverable ladder.** Design note → spring-slider (single degree of
freedom, validated against analytic steady state and the a/b stability
boundary — also the required pedagogical example) → single planar fault
quasi-dynamic port with validation → triangular meshes, checkpointing,
event catalogs → JAX/differentiable integration only after the CPU
reference is trusted (which also needs F9). Large effort spread over many
increments; needs your priority decision (standing question 4).

## C7. Layered elastic half-space engine

**What.** Every current engine assumes a uniform elastic half-space.
Real crust is layered, and layering measurably changes surface
displacements for deep or large sources. Options: (a) an in-house
propagator-matrix implementation (self-contained but numerically
delicate: wavenumber integration, high-contrast stability); (b) wrap an
established code (EDKS, PSGRN/PSCMP) behind an optional dependency
(Fortran toolchain and licensing friction); (c) precomputed
Green's-function tables plus interpolation, with the table generator as
the optional external step.

**Proposal.** Prototype (c)'s table format first — it also serves (b) —
and survey licenses before wrapping anything. Begin with the well-bounded
case: surface displacements from rectangular sources over horizontal
layers, cross-validated against published half-space-versus-layered
comparisons; strain and internal fields later. The engine's material
description extends `ElasticMedium` (the 0.4 decision anticipated this).
Viscoelastic and poroelastic engines stay out of scope until the engine
contract can represent *time*. Large effort, the biggest unknowns on this
menu; sequence last among the engines. Needs your dependency-tolerance
decision (standing question 3).

## C8. Mesh quality, refinement, and transfer

**What.** Quality metrics (aspect ratio, skewness, size grading) on
`Mesh`/`Fault.validate()`; adaptive refinement driven by data sensitivity
(the resolution diagonal from `invert.model_resolution` as the driver);
and transfer operators (patch-to-patch interpolation matrices) so a slip
model moves between meshes with stated conservation properties
(moment-preserving versus value-interpolating — both, named).

**Stance.** The refinement loop is a research feature with real UX risk:
prototype it as an `examples/` study before committing to any API. The
quality metrics and transfer operators are independently useful and can
land first. Medium effort overall.

## C9. Elastic-parameter sensitivity

**What.** Derivatives of predictions with respect to μ and ν via the
differentiable kernels (F9), delivering the Chapter 13 misspecification
story with derivatives instead of finite differences, and honest
uncertainty inflation for slip models when the elastic structure is
imperfectly known. Joint inversion for elastic structure stays out until
this sensitivity layer exists. Small-to-medium effort once F9 ships.

## C10. Kinematic time-dependent slip

**What.** Slip histories (afterslip decay, slow-slip events) from
multi-epoch data with temporal basis functions. Explicitly last: it
requires the multi-epoch metadata bridge (F5) and temporal bases that do
not exist yet. Do not design it now; the one recorded constraint is that
F5's epoch metadata must not preclude a time-basis-expansion `G` later.

---

## Standing questions that need user decisions

These are decisions, not experiments — work on the affected items should
not start until they are made:

1. **Reference-source location (C6).** Where should recovered
   `unicycle`/`stress-shadows` sources live — vendored under `geometry/`
   with the other reference code, or fetched on demand like the skipped
   `Fault.load` test data?
2. **The `Source` object (C3).** Is a volcanic `Source` domain object
   acceptable growth of the public object budget, or should volcanic
   inversion stay functions-only until demand is demonstrated?
3. **Fortran tolerance (C7).** In-house layered implementation versus
   wrapping an external code — how much build/dependency friction is
   acceptable in an optional extra?
4. **Cycle-modeling priority (C6).** Is earthquake-cycle modeling
   actually the highest-value capability for the package's users, or
   should the engines (C2/C3) and inference (C1) items lead? The old
   roadmap ordered cycle modeling first; this menu deliberately does not.
