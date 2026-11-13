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
  and `LinearSystem` as the one reusable inversion object. The engine
  registry, API-tier stability map, layering test, and deprecation policy
  were removed. The pass also found and fixed reversed row depths in
  `Fault.planar` (see `CHANGELOG.md`).

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

## Simplification backlog

What the November 2026 pass left, in the order a student is likely to
meet it. Each item is independent.

| # | Item | Why it matters | Effort |
|---|---|---|---|
| S1 | One spelling for the local origin | `ref_lat`/`ref_lon` and `frame=` are both accepted by `Fault.from_triangles`, `geometry_search`, `bayes`, and `mesh`; `LocalFrame(lat, lon)` is positional and latitude-first. Settle on `frame=LocalFrame(lon=, lat=)`. | Small-medium |
| S2 | One Green's assembly path | `greens.displacement_greens` loops over patches in NumPy and has a separate batched JAX branch; the batched kernel already broadcasts in NumPy. Use it for both, and give the public functions keyword `lon`/`lat`. Needs the `exact` golden tests re-baselined. | Medium |
| S3 | One projection for rectangles and triangles | Rectangles use a per-patch tangent plane, triangles one shared `LocalFrame`; `transforms.translate_flat` is a third. A shared frame would make `Fault.corners` exact and remove a class of small inconsistencies. **Numerical** change. | Medium |
| S4 | Slimmer `InversionResult` | More than thirty fields; group provenance (backend, precision, hashes, bounds, constraints) into one nested record so the fields a student prints stand out. | Small |
| S5 | Simpler result files | `invert/_io.py` is a 600-line versioned, checksummed schema with migrations. With no released files to protect, a plain `.npz` plus JSON metadata would do. | Small |
| S6 | Cache off by default | The disk cache for `G` writes `.geodef_cache/` in the working directory without being asked. Make it opt-in (`geodef.cache.enable()`), and show it in Chapter 02. | Small |
| S7 | Kernel subpackage | Move `okada85`, `okada92`, `tri`, and `okada` under `geodef.kernels` so the top-level module list reads as a table of contents. | Small (mechanical) |
| S8 | Merge `geometry` and `transforms` | Two modules for coordinates; `LocalFrame` could own the conversions a student needs. | Small |
| S9 | Tutorial data setup in chapters 11-14 | Still built from `fault.displacement` plus per-component noise; moving to `synthetic` changes their random streams and therefore their printed numbers. | Small |
| S10 | Dataset-valued predictions | `result.prediction()` returns arrays keyed by name; returning datasets (`gnss.with_obs(...)`) would let `plot.vectors(observed, predicted)` drop its manual slicing. | Small-medium |

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

1. **S1, S6, S4, S5** — the small simplifications, while the API is
   still free to change.
2. **F1 remainder** (checkerboard/spike/compare) and **S9/S10**, which
   finish the tutorial cleanup.
3. **S2 then S3** — the Green's assembly and projection unification,
   each with its golden data re-baselined in one commit.
4. **F2** (coupling) with its interseismic example.
5. **A3 → F3 → F4 → F5** — the noise and whitening layer and the
   features on it.
6. Then by demand: F6, F8, A6 → C1, and new engines.

---

## Release discipline

- **v0.2:** after the simplification backlog's small items, so the
  course, docs, and vocabulary ship together.
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
