# PLAN.md — GeoDef Roadmap

GeoDef should be both a teaching library and a research tool: a student
expresses a geophysical problem with a few memorable names and can read
the code underneath, while an expert can still reach the Green's
matrices, covariance operators, autodiff kernels, constrained solvers,
and Bayesian posteriors.

This document is forward-looking. Git history, tests, and `CHANGELOG.md`
record what shipped; `CLAUDE.md` holds the working rules and the API
conventions.

---

## Where things stand

- **Physics and domain layer.** Rectangular Okada 1985/1992 and
  triangular Nikkhoo & Walter dislocation engines (surface and internal
  displacement and strain); `Fault`, `GNSS`, `InSAR`, `Vertical`;
  `LocalFrame`; `ElasticMedium`; rectangular and triangular mesh
  construction; Euler poles; cached Green's assembly.
- **Inversion.** Weighted, bounded, constrained, fixed-direction, and
  regularized linear slip inversion; L-curve, ABIC, cross-validation;
  covariance, resolution, per-dataset diagnostics; versioned result files.
- **Research surface.** Optional JAX backend; differentiable forward
  models and gradient-based planar geometry search; collapsed and
  positivity-aware Bayesian inference with NUTS sampling.
- **Learning experience.** A tested quickstart, workflow and decision
  guides, glossary, a fifteen-chapter executed course with solutions, a
  plot gallery, and real-data examples.
- **Simplification pass (November 2026).** One constructor per dataset,
  keyword `lon`/`lat` and `strike_slip`/`dip_slip` everywhere,
  `DataSet.synthetic`, public `Fault` geometry with a single `corners`
  accessor, assessment as methods on a result that remembers its problem,
  and `LinearSystem` as the one reusable inversion object. Then the
  ten-item backlog: an opt-in cache, one origin spelling
  (`frame=LocalFrame(lon=, lat=)`), coordinates in `transforms`, the
  kernels in `geodef.kernels`, one map projection for rectangles,
  triangles, and stations, one batched Green's assembly path, a slim
  `InversionResult` with `details`, plain `.npz` result files,
  dataset-valued predictions, and every chapter on `synthetic`. The engine
  registry, API-tier stability map, layering test, and deprecation policy
  were removed. The pass found and fixed three physics errors (reversed
  planar row depths, per-patch projections that kept large faults from
  being planar, and normal-positive dip slip on generated triangle meshes);
  see `CHANGELOG.md`.

---

## Product principles

Acceptance criteria for every change.

1. **One obvious way.** Each task has one spelling, used by the code,
   the tutorials, and the docs. A second path needs a reason a student
   would accept.
2. **Geophysical names before array conventions.** `strike_slip`,
   `dip_slip`, named datasets, and coordinate frames by name. Blocked
   vectors and stacked rows appear where the linear algebra is the lesson.
3. **Units and frames are never implicit.** SI by default; every non-SI
   parameter says so; coordinates are keyword `lon=`/`lat=`.
4. **Readable to the bottom.** A student can follow a call from
   `geodef.solve` to the kernel without detouring through registries,
   protocols, or compatibility layers.
5. **Reveal complexity progressively.** Sensible defaults for a novice;
   the Green's matrix, noise model, regularizer, and solver stay reachable.
6. **Teach the model, including its limits.** Covariance, resolution,
   geometry uncertainty, and prior sensitivity are visible.
7. **Optional power stays optional.** JAX, blackjax, Cartopy, and mesh
   generation never burden the base install.
8. **Physical tests.** Pin behavior with invariants (subdivision,
   superposition, recovery of a known model), not with a second copy of
   the formula.
9. **No compatibility machinery before 1.0.** Rename everywhere and
   record it in the changelog; no aliases, tiers, or deprecation cycles.

---

## The roadmap menus

Remaining feature work is organized as three menus. Items are
independent unless a dependency is stated; each menu opens with a
prioritized table.

- **`plans/ARCHITECTURE.md` — internal structure.** Kernel adapters
  (A2), whitening and operator-valued linear algebra for realistic data
  sizes (A3), solver conditioning (A4), JAX backend-state hygiene (A5),
  sampler-independent results (A6), boundary contracts (A7), the
  result-schema rule (A9).
- **`plans/FEATURES.md` — improvements to existing capabilities.**
  Remaining synthetic-test helpers (F1), first-class interseismic
  coupling (F2), noise modeling (F3), nuisance parameters (F4), data
  preprocessing (F5), large-problem solve paths (F6), sweep acceleration
  (F7), guided Bayesian geometry inference (F8), differentiable
  strain/stress (F9), examples and a documentation site (F10).
- **`plans/CAPABILITIES.md` — new capabilities.** Tempered SMC and model
  comparison (C1), the Meade engine (C2), volcanic sources (C3), tensile
  slip (C4), multi-fault systems (C5), earthquake-cycle modeling (C6), a
  layered half-space (C7), mesh refinement (C8), elastic-parameter
  sensitivity (C9), kinematic slip (C10). Several need a user decision
  first; those questions are collected at the end of that document.

### A recommended near-term sequence

1. **F1 remainder** (checkerboard and spike patterns, `result.compare`),
   which finishes the tutorial cleanup, then tag **v0.2**.
2. **F2** (coupling) with its interseismic example.
3. **A3 → F3 → F4 → F5** — the noise and whitening layer and the
   features on it.
4. Then by demand: F6, F8, A6 → C1, and new engines.

New simplification candidates go in this section as they are found.

---

## Release discipline

- **v0.2:** after the F1 remainder, so the simplified API, course, and
  docs ship together.
- **Later 0.x releases:** menu items in the order chosen; each release
  batches its result-file changes.
- **v1.0:** after the high-priority menu items and human testing with
  students. This is the point at which a compatibility promise begins.

For each user-facing change: tests that pin the physics, an executed
tutorial or example on the public path, updated reference docs, and a
changelog entry with a before/after.

---

## Enduring design conventions

- Public Cartesian coordinates are East, North, Up; depth is positive
  down; the fault dips to the right of strike; row 0 of a planar grid is
  the shallowest.
- Lengths, displacements, and slip are in meters; stress in Pa; angles in
  degrees unless a name says otherwise.
- The linear-algebra slip view is blocked `[:N]` strike-slip, `[N:]`
  dip-slip. Named views are preferred in everyday code.
- NumPy is the default runtime; JAX is optional.
- Caching may accelerate pure computations but must never change
  results; cache keys include every input that affects the result.
- Keep this plan forward-looking: when work ships, replace its checklist
  with a line under "Where things stand".
- Do not add `Co-Authored-By` trailers; AI attribution is maintained in
  `README.md`.
