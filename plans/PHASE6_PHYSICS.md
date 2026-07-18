# Phase 6 Plan (Tentative) — New Physics and Research Capabilities

This document is a **tentative design sketch for PLAN.md Priority 6**:
quasi-dynamic earthquake-cycle modeling (6.1), additional Green's engines
(6.2), and richer fault/slip models (6.3). Unlike the Phase 3–5 plans, this
phase is deliberately speculative: several items need prototypes,
literature checks, or user decisions before a committed plan is honest.
Open questions are listed per item with a proposed resolution path; expect
this file to be rewritten at least once before work begins.

> **Status: drafted 2026-07 — exploratory.** Hard prerequisites: the
> Phase 3.3 engine capability contract (every new engine targets it), the
> Phase 5.1 differentiable stress kernels (6.1's JAX path), and — for
> 6.1's port — the `related/stress-shadows/unicycle/` reference sources,
> which are **not present in a fresh clone** (only `related/shakeout_v2/`
> is); recovering them is a setup task, not an afterthought.
>
> **Lifecycle.** Transient. 6.1's first committed artifact is itself a
> design note that supersedes §2 of this file.

---

## 1. Ground Rules (inherited, restated)

- New physics must ride the engine/operator interfaces (3.3), not add
  special cases to beginner-facing code; `Fault.planar` + `geodef.solve`
  never learn about friction laws or layered media directly.
- Everything is an optional extra (`geodef[cycle]`, engine-specific
  extras); the base install and the tutorial course are untouched.
- Every engine: cite the equations, preserve a reference implementation,
  cross-validate published cases, declare capabilities and coordinate
  conventions, and show one end-to-end example through the same
  high-level workflow (PLAN.md 6.2 acceptance list).

---

## 2. Item 6.1 — Quasi-Dynamic Earthquake-Cycle Modeling

### Intended shape

An optional `geodef.cycle` module: rate-and-state friction on the existing
rectangular/triangular fault meshes, driven by the elastic stress kernels
GeoDef already computes (`fault.stress_kernel`), with quasi-dynamic
radiation damping in place of full inertia. CPU/NumPy reference first;
differentiable JAX integration only after the reference is trusted.

### Committed first step: the design note

Per the roadmap, the first deliverable is a standalone design note (it
replaces this section) fixing: scope, state variables, sign/unit
conventions, validation targets, and the boundary between static GeoDef
values and `geodef.cycle`. The note must answer the open questions below —
several require small experiments, flagged **[experiment]**.

### Deliverable ladder (tentative)

1. Design note (above).
2. **Spring-slider** (single degree of freedom): friction-law functions
   (aging law first), adaptive ODE integration, event detection; validated
   against the analytic steady state, the a/b stability boundary, and
   published limit-cycle behavior. This is also the pedagogical example
   the roadmap requires before any fault-system run.
3. **Single planar fault, quasi-dynamic**: stress-kernel-driven evolution
   ported from `related/stress-shadows/unicycle/` and independently
   validated (SEAS benchmarks, below).
4. Triangular meshes; checkpoint/restart files; energy/moment
   diagnostics and event catalogs.
5. JAX/differentiable integration (diffrax or hand-rolled), only after 3–4
   are trusted.

### Open questions (with resolution paths)

1. **Stress-kernel completeness.** Cycle modeling needs shear traction
   resolved in the rake direction *and* (for some laws) normal-stress
   changes. What exactly does today's `fault.stress_kernel` return, and is
   a normal-traction kernel needed for v1? → **[experiment]** audit the
   kernel against `unicycle`'s and decide whether v1 fixes normal stress
   (common simplification) — record in the design note.
2. **Friction-law scope**: aging law only, or aging + slip law from the
   start? Proposal: aging only for validation, slip law as the second
   registered law once the callable interface exists. → design-note
   decision, cheap to defer.
3. **Validation targets**: adopt SEAS benchmark problems (BP1-QD is the
   natural first target) vs replicating `unicycle` runs? Proposal: both —
   `unicycle` for port equivalence, one SEAS problem for community-facing
   validation. → literature check + **[experiment]** confirm a BP1-scale
   run is tractable in NumPy at acceptable resolution/wall-clock.
4. **Integrator**: `scipy.integrate.solve_ivp` (LSODA/BDF — the system is
   stiff between events) vs a custom adaptive RK like `unicycle`'s.
   → **[experiment]** benchmark both on the spring-slider and a small
   fault; stiffness handling and event-detection hooks decide it.
5. **Units and time**: SI seconds internally with year-based conveniences
   at the interface, or years throughout? Interseismic APIs (4.4) speak
   m/yr. → design-note decision; must be settled before any code.
6. **State/checkpoint format**: does checkpointing justify a state record
   class (the roadmap's "only if invariants require one")? → defer until
   step 3 shows what a restart actually needs.
7. **Source recovery**: `related/stress-shadows/` is absent from fresh
   clones. Confirm licensing/attribution (the author's own code, per 0.3)
   and either vendor the needed `unicycle` files into `geometry/` or
   document the retrieval step. → user decision on where the reference
   sources should live.

---

## 3. Item 6.2 — Additional Green's Engines

Each engine is a Phase 3.3 `EngineSpec` registration; the first two are
also the acceptance test for making the registry public (two external
engines rule).

### 6.2a Meade (2007) triangular dislocations

- Purpose: an *independent cross-check* on the Nikkhoo & Walter `tri`
  engine, valuable for research verification.
- Shape: same capability surface as `tri` (surface + internal
  displacement/strain), same `Fault.from_triangles` geometry — a pure
  second engine, `engine='meade'`.
- Open questions: port from Meade's published Matlab (license: check
  header terms); known artifact cases near element edges differ from
  Nikkhoo — the cross-validation must *document* disagreements, not hide
  them. → **[experiment]** port a minimal displacement path and compare
  against `tri` on the existing reference geometries before committing to
  a full port.

### 6.2b Point and compound sources (volcanic deformation)

- Candidates: Mogi point source, McTigue finite sphere, compound
  dislocation model (CDM, Nikkhoo et al. 2017), point CDM.
- **The real design problem is not the kernels** (they are short); it is
  that these sources are not `Fault`s: no patches, no slip vector, no
  strike/dip — the unknowns are pressure/volume change and geometry.
  Options: (a) generalize `Fault` (rejected on sight — muddies the core
  class); (b) a parallel `Source` domain object with its own small
  forward/inverse surface; (c) engine-level functions only, no domain
  object, inversion via the generic operator interfaces. → needs a design
  decision with user input; prototype (c) first since it adds no public
  classes and the object budget demands the justification.
- Data side is already sufficient: GNSS/InSAR/Vertical datasets and the
  noise/nuisance machinery (Phase 4) apply unchanged.

### 6.2c Layered (elastic) half-space displacements

- Approach options: (a) in-house propagator-matrix implementation
  (self-contained, significant numerical care: wavenumber integration,
  stability at high contrast); (b) wrap an established code (EDKS,
  Wang's PSGRN/PSCMP) behind an optional dependency (build/licensing
  friction, Fortran toolchains); (c) precomputed Green's-function tables
  + interpolation, with the table generator as the optional external
  step. → **decision needed**; proposal: prototype (c)'s table format
  first since it also serves (b), and survey licenses before any wrap.
- Architecture note: a layered engine's `EngineSpec` must carry material
  parameters richer than `ElasticMedium(mu, nu)` — the 0.4 decision to
  make `ElasticMedium` the extensible home for this is the hook
  (`LayeredMedium` extending it, per PLAN.md 0.4's forward reference).
- Begin with a well-bounded use case: surface displacements from
  rectangular sources over horizontal layers, cross-validated against
  published half-space-vs-layered comparisons; strain/internal fields
  later. Viscoelastic/poroelastic remain out of scope until the engine
  contract can represent *time* — explicitly deferred, per the roadmap.

---

## 4. Item 6.3 — Richer Fault and Slip Models

### Tensile/opening components

- The blocked `2N` convention (`[:N]` strike, `[N:]` dip) is an enduring
  design convention; a third component must be **opt-in without
  disturbing it**. Sketch: `components='full'` (or `'tensile'`) yields a
  `3N` system with the third block opening; `slip.pack`/`unpack` gain an
  optional third argument/return; result views add `opening`; moment
  semantics (tensile moment tensor vs shear moment) made explicit.
- Open question: every consumer of the blocked layout (regularization
  block structure, plotting, bounds expansion, result schema) is touched
  — enumerate them and confirm the default `2N` path stays literally
  unchanged (the tutorials' contract). → mini design note + audit before
  implementation.

### Multiple faults / segments

- Solve over a list of `Fault`s with named model partitions (the
  model-side analogue of 1.5's dataset names + `dataset_slices`):
  per-fault slip views, per-fault moment, one joint `G`.
- Continuity/boundary constraints across segment joins expressed through
  the existing `constraints=(C, d)` machinery, with a helper that builds
  the continuity rows from segment adjacency.
- Open question: is a `FaultSystem` container justified (object budget)
  or is `solve([fault_a, fault_b], ...)` + named partitions enough?
  Proposal: the list form, no new class, until cross-call invariants
  (shared frames, adjacency) prove otherwise.

### Mesh quality, refinement, and transfer

- Quality metrics (aspect ratio, skewness, size grading) on `Mesh` /
  `Fault.validate()`; adaptive refinement driven by data sensitivity
  (resolution diagonal from `invert.model_resolution` as the driver);
  transfer operators (patch-to-patch interpolation matrices) so a slip
  model moves between meshes with stated conservation properties
  (moment-preserving vs value-interpolating — both, named).
- Open question: refinement loop = research feature with real UX risk;
  → prototype as an `examples/` study before any API.

### Elastic-parameter sensitivity

- `d(prediction)/dν`, `d(prediction)/dμ`-family sensitivity via the
  Phase 5.1 differentiable kernels; deliverable is the Chapter 13
  misspecification story with derivatives instead of finite differences,
  and honest uncertainty inflation for slip models. Joint inversion for
  elastic structure stays out until this sensitivity layer exists (per
  the roadmap ordering).

### Kinematic time-dependent slip

- Explicitly last: requires multi-epoch datasets and temporal bases
  (Phase 4.3's deferred bridge). Do not design it here; record only the
  constraint that 4.3's epoch metadata must not preclude a
  time-basis-expansion `G` later.

---

## 5. Sequencing Sketch (tentative)

1. 6.1 design note (+ source recovery) — can start any time; the
   spring-slider needs nothing from Phases 3–5.
2. 6.2a Meade port prototype — after the 3.3 registry exists.
3. 6.1 fault-scale port + validation — after 6.1's note settles the
   integrator/units questions.
4. 6.2b point-source prototype (option c) — independent; informs the
   `Source` design decision.
5. 6.3 tensile audit + multi-fault helpers — after the Phase 3 splits
   (touching `_solvers`/`_system` is far safer post-split).
6. 6.2c layered engine — last; largest unknowns, needs the dependency
   decision.

## 6. Standing Open Questions (need user input, not experiments)

1. Where should recovered `unicycle`/`stress-shadows` reference sources
   live — vendored under `geometry/`, or fetched-on-demand like the
   skipped `Fault.load` test data? (§2 Q7)
2. Volcanic sources: is a `Source` domain object acceptable growth of
   the public object budget, or should volcanic inversion stay
   functions-only until demand is demonstrated? (§3 6.2b)
3. Layered engine: in-house implementation vs wrapping an external code
   — tolerance for Fortran build dependencies in an optional extra?
   (§3 6.2c)
4. Is earthquake-cycle modeling (6.1) actually the highest-value Phase 6
   item for the package's users, or should engines (6.2) lead? The
   roadmap orders 6.1 first; the sequencing above lets them interleave —
   confirm priority before committing the design note.
