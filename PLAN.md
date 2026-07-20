# PLAN.md — GeoDef Roadmap

GeoDef's goal is not a choice between a small teaching library and a
powerful research package. It should be both: a novice should be able to
express a geophysical problem with a few memorable functions and durable
domain values, while an expert can still reach the Green's matrices,
covariance operators, autodiff kernels, constrained solvers, and Bayesian
posteriors underneath.

This roadmap is forward-looking. Git history, tests, and the module
documentation preserve the details of completed work; this document
records what is done at a baseline level, and points to the three menu
documents that describe what remains.

**Read `PYTHON.md` before editing any code.** Use red/green TDD, update
relevant documentation with every public change, and commit each
independently useful unit with its relevant tests passing. Run the full
routine suite when a pull request or major change is wrapping up.

---

## Current foundation

The original roadmap's Priorities 0–3 are complete apart from a few
leftovers that have been folded into the menus below. The baseline:

- **Physics and domain layer.** Rectangular Okada 1985/1992 and triangular
  Nikkhoo & Walter dislocation engines with surface and internal
  displacement/strain paths; `Fault`, `GNSS`, `InSAR`, and `Vertical`
  domain objects; `LocalFrame` coordinate frames; `ElasticMedium` as the
  single home for material parameters; rectangular and triangular mesh
  construction; Euler-pole fitting; Green's assembly with provably
  complete disk-cache keys.
- **Inversion.** Weighted, bounded, constrained, fixed-direction, and
  regularized linear slip inversion; L-curve, ABIC, cross-validation,
  uncertainty, resolution, and per-dataset diagnostics; conditioning
  reports; a versioned, migrating result-file schema.
- **Consistency and trust (P0).** One frozen regularization convention;
  unambiguous `chi2`/`reduced_chi2` vocabulary; a single conventions
  reference (`docs/conventions.md`, including the backslip/coupling
  convention); centralized physical validation with `.validate()` reports;
  LICENSE, CITATION, CI from scratch, `py.typed`, single-sourced version,
  tested install tiers, changelog, and deprecation policy.
- **The everyday API (P1).** An enforced public object budget; specific
  function names on module paths (`invert.solve`, `greens.matrix`,
  `slip.from_rake`, ...); keyword-only geometry and data constructors;
  named result views; friendly data constructors with table ingestion and
  first-class dataset names; the pre-documentation breaking-change batch
  (name collisions, meters everywhere, keyword-only solve).
- **Learning experience (P2).** A five-minute quickstart; workflow map,
  glossary, and decision guides; a fifteen-chapter executed course
  (00–14) with objectives, exercises, and solution notebooks; uniform,
  reproducible examples; usability metrics.
- **Internals (P3).** A declared, test-enforced layer structure and API
  stability map (`docs/api_stability.md`); `invert`/`bayes`/`plot` split
  into packages and `fault` I/O extracted, all behind unchanged public
  names; a private engine registry with capability declarations and typed
  capability errors; property-based numerical contracts; an exact/
  physical/benchmark test taxonomy; an on-demand benchmark harness.
- **Research surface.** NumPy default with optional JAX; float32/float64
  control; differentiable rectangular and triangular displacement models
  and gradient-based planar geometry search; collapsed and
  positivity-aware Bayesian inference (`RectPosterior`, `SlipPosterior`,
  `TriWarp`, `TriPosterior`), NUTS sampling, convergence diagnostics,
  conditional slip draws, and prediction.

---

## Product principles

These are acceptance criteria for every roadmap item.

1. **One obvious beginner path.** The first successful forward model and
   inversion should use a small, stable vocabulary. Alternative solvers and raw
   matrices remain available, but should not be prerequisites.
2. **Geophysical names before array conventions.** Public results should expose
   `strike_slip`, `dip_slip`, station predictions, and coordinate frames by
   name. Blocked vectors and stacked rows remain available as explicit linear-
   algebra views.
3. **Units and frames are never implicit.** SI units remain the default. Every
   non-SI parameter carries a suffix, and every local coordinate array is tied
   to a declared origin and axis convention.
4. **Reveal complexity progressively.** A novice can use sensible defaults; a
   researcher can inspect or replace the Green's matrix, noise model,
   regularizer, parameterization, backend, and solver.
5. **Teach the model, including its limits.** Documentation explains the
   assumptions behind a result and makes covariance, resolution, geometry
   uncertainty, and prior sensitivity visible.
6. **Optional power stays optional.** JAX, BlackJAX, Cartopy, mesh generation,
   and future cycle/layered-earth dependencies must not burden the base install.
7. **Stable public surface, replaceable internals.** Refactors preserve public
   imports and numerical behavior. Deprecations include warnings, migration
   examples, and at least one minor-release transition period.
8. **Array transparency without array traps.** NumPy arrays remain accepted and
   returned where natural, but common shape/order mistakes should be prevented
   by named accessors and precise validation.
9. **Functions are the default abstraction.** Add a public class only when it
   owns durable state, preserves an invariant that functions cannot preserve,
   or amortizes expensive preparation across calls. Parameter bundles and
   transient array views stay as keyword arguments, arrays, and module
   functions.

---

## The roadmap menus

The remaining work is organized as three menus rather than sequential
phases. Items are piecemeal: each can be picked independently, dependencies
are stated per item, and each menu opens with a prioritized summary table
and a recommendation.

### `plans/ARCHITECTURE.md` — internal structure

Work on how the code is built, with plain-language discussion of why each
item matters. Highlights: finishing the engine registry (A1), the kernel
adapter layer (A2), whitening and operator-valued linear algebra — the
key enabler for realistic data sizes (A3), solver conditioning defaults
(A4), JAX backend-state hygiene (A5), the sampler-independent result
contract (A6), boundary contracts (A7), the top-level export trim that
must precede any public v0.2 tag (A8, shipped), and the standing
schema-bump rule (A9).

### `plans/FEATURES.md` — improvements to existing capabilities

Better versions of things GeoDef already does. Highlights: synthetic-test
helpers and the scenario builder (F1), first-class interseismic coupling
(F2), noise modeling tools (F3), nuisance parameters (F4), data
preprocessing algebra (F5), large-problem solve paths (F6), sweep
acceleration (F7), guided Bayesian geometry inference (F8),
differentiable strain/stress kernels (F9), and the real-data examples,
golden workflows, and documentation site (F10).

### `plans/CAPABILITIES.md` — new capabilities

Things GeoDef cannot do at all today. Highlights: tempered SMC and model
comparison (C1), the Meade cross-check engine (C2), volcanic sources
(C3), tensile slip (C4), multi-fault systems (C5), earthquake-cycle
modeling (C6), a layered half-space engine (C7), mesh refinement (C8),
elastic-parameter sensitivity (C9), and kinematic time-dependent slip
(C10). Several items need a user decision first; those questions are
collected at the end of that document.

### Where the old priorities went

Nothing from the retired phase plans (`plans/PHASE3–6`) was dropped; the
still-open items map as follows:

| Old item | New home |
|---|---|
| 2.2 scenario builder, synthetic helpers | F1 |
| 2.2 top-level export reduction | A8 (shipped) |
| 2.3 real-data examples, docs site; 2.4 golden workflows, observation sessions | F10 |
| 3.2 kernel adapters | A2 |
| 3.3 registry completion, typed contracts | A1 |
| 3.4 boundary contracts; solver benchmarks | A7; A4 |
| 4.1 noise/whitening | A3 (plumbing) + F3 (tools) |
| 4.2 operators, iterative solvers, memory | A3/A4 (plumbing) + F6/F7 (paths) |
| 4.3 nuisance, quadtree, splits, multi-epoch bridge | F4, F5 |
| 4.4 interseismic coupling | F2 |
| 5.1 batched sweeps, differentiable strain, backend hygiene | F7, F9, A5 |
| 5.2 guided geometry inference | F8 |
| 5.3 SMC, result contract | C1, A6 |
| 6.1 cycle modeling | C6 |
| 6.2 Meade, volcanic, layered engines | C2, C3, C7 |
| 6.3 tensile, multi-fault, refinement, sensitivity, kinematic slip | C4, C5, C8, C9, C10 |

`tutorials/OUTLINE.md` remains in place until F1 ships (it holds the
settled `geodef.synthetic` design); it is then retired to git history
like the phase plans.

### A recommended near-term sequence

The menus are menus — but if asked for a default order:

1. **F1** (synthetic helpers): small, closes out the Priority 2 leftovers
   alongside the export trim (A8, already shipped), enabling a public
   v0.2 tag.
2. **F2** (coupling): strategic, convention already settled, no
   prerequisites; deliver the interseismic example (F10) with it.
3. **A5 + A7** (backend capture; boundary contracts): small protective
   items, any time.
4. **A3 → F3 → F4 → F5**: the noise/whitening layer and the features on
   it, delivering the earthquake example (F10) at the end.
5. Then by demand: F6 when data sizes require it, F8 for research users,
   A6 → C1 and A1 → C2 as the first new capabilities.

---

## Release discipline

- **v0.2:** tag after step 1 above — the course, docs, and final public
  vocabulary shipped together.
- **Later 0.x releases:** menu items in the order chosen, each release
  batching its result-schema changes into one migration (rule A9).
- **v1.0:** only after the menus' high-priority items are complete and
  the resulting public workflows have gone through human testing (F10's
  observation sessions). New engines or a large feature count do not by
  themselves justify 1.0.

For each user-facing change, require:

- a before/after golden workflow showing fewer manual transformations;
- unit, integration, documentation, and backwards-compatibility tests;
- an executed tutorial or example using the public path;
- numerical equivalence against the current implementation where semantics
  are unchanged;
- release notes with migration examples and explicit non-goals.

---

## Enduring design conventions

- Public Cartesian coordinates are East, North, Up; public depth is positive
  down. Kernel-native conventions are converted at adapters.
- Base geometry lengths, displacements, and slip use meters; stress uses Pa;
  angles use degrees unless an API name explicitly says otherwise.
- The expert linear-algebra slip view remains blocked as
  `[:N]` strike-slip and `[N:]` dip-slip. Named domain views are preferred in
  everyday code.
- NumPy remains the default runtime and JAX remains optional.
- Hash-based caching may accelerate pure computations but must never change
  results or hide problem provenance. Cache keys include every input that
  affects the result — geometry, observation locations, material parameters,
  precision, and a kernel version — or the computation is not cached.
- `PLAN.md` is forward-looking. When work ships, replace detailed checklists
  with a concise baseline entry or release note rather than accumulating an
  implementation diary here.
- Update this plan (and the affected menu document) in the same logical
  commit when scope or a settled design decision changes.
- Do not add `Co-Authored-By` trailers. AI model attribution is maintained in
  `README.md`, `AGENTS.md`, and `CLAUDE.md` according to repository policy.
