# PLAN.md — Forward-Looking Roadmap for GeoDef

**GeoDef v1.0 has shipped.** The runtime library, the ten-part tutorial course,
the per-module documentation, and clean `ruff`/`mypy`/warning-free tooling are
all complete. This document is now **forward-looking only**: it records where the
project is headed, not what has already been done.

**Read `PYTHON.md` before editing any code.**

---

## Guiding principles

GeoDef is meant to stay *simple, readable, and good for learners* while remaining
capable for research. Every item below should be weighed against that: prefer a
small, well-tested, well-documented addition over a large framework. Optional and
heavy dependencies (GPU stacks, cycle solvers, extra Green's engines) must stay
**optional** and never burden the base install or the teaching path.

Each roadmap item, when started, becomes its own `PLAN.md` step with concrete
sub-tasks, tests, and docs, following the same red/green TDD workflow and small-
commit discipline used to reach v1.0.

---

## 1. GPU / autodiff accelerator (headline effort — IN PROGRESS)

**Goal.** A drop-in accelerated backend for the two hot loops — Green's-function
assembly and the inverse solve — using a **JAX** array backend with automatic
differentiation. This is the single highest-leverage extension: it turns the
dense, embarrassingly-parallel Okada/triangular kernels and the linear algebra
of inversion into XLA-compiled (and, where available, GPU) work, and it makes
the *nonlinear* geometry search (tutorial 10) gradient-based and fast.

Why it fits GeoDef: the physics kernels are pure, vectorized array math with no
Python-loop dependence, and the inverse problem is a differentiable pipeline from
geometry → `G` → slip → predicted data → misfit. That is exactly the shape
autodiff rewards.

### Decisions (settled 2026-07)
- **JAX, not CuPy.** Phases 2–3 require autodiff, which CuPy lacks; JAX also
  runs on plain CPUs (XLA JIT), NVIDIA, and AMD. Expectation to document
  honestly: on Apple-silicon laptops `jax-metal` is experimental and lacks
  float64, so the realistic laptop win is JIT-compiled CPU + `vmap`, not GPU
  offload — benchmarks should present it that way.
- **Backend selection:** a global switch, `geodef.set_backend("jax")` /
  `("numpy")`, defaulting to NumPy, plus a `GEODEF_BACKEND` environment
  variable. No per-call `backend=` arguments.
- **Precision:** float64 default (set `jax_enable_x64` before any JAX op);
  a documented **opt-in** float32 mode validated with looser tolerances.
- **Phase 3 scope:** accelerate the unconstrained (`wls`) solve and batched
  hyperparameter sweeps first; `nnls`/`bounded_ls`/`constrained` stay on the
  SciPy CPU path (no `jaxopt`/`optax` dependency for now).
- **Phase 2 scope:** differentiate **both** the rectangular Okada and the
  triangular (Nikkhoo & Walter) engines from the start, so the `tri` kernel
  refactor is designed for tracing/gradients from day one.

### Phase 1 — Backend abstraction (CPU-only, no behavior change)
- [x] `geodef.backend`: `set_backend()`/`get_backend()`, `GEODEF_BACKEND` env
  var, array-namespace resolution (NumPy default, JAX when installed and
  selected), float64 default with opt-in float32.
- [x] Trace-safe kernel refactor (pure NumPy, behavior-preserving): rewrite
  `tri.trimodefinder`'s `flatnonzero` + fancy-index assignment into
  `where`/mask form via `backend.masked_eval` (NumPy gathers/scatters as
  before; JAX evaluates full-size and selects with `where`). Validated
  against the Matlab reference `.npz` files at existing tolerances.
- [x] Route the `okada85` and `tri` math through the backend namespace
  (`backend.xp` proxy); NumPy stays the default everywhere — invisible to
  existing users and the tutorials.
- [x] Cross-validate JAX output against the existing Matlab reference `.npz`
  files to the same tolerances the CPU engines meet
  (`tests/test_backend_kernels.py`, skipped when JAX is absent).
- [x] Batched Green's assembly for the JAX path: `displacement_greens`,
  `strain_greens` (surface okada85 path), and `strain_greens` at depth
  (DC3D path, used by `Fault.stress_kernel`) broadcast their kernels over
  a leading patch axis and JIT-compile the batched call; the NumPy loop
  paths are untouched. Measured: a 100x100 self-stress kernel assembles
  in ~12 ms steady-state on JAX/CPU vs ~550 ms on NumPy (~46x), after a
  one-time ~6 s DC3D compile per problem shape. Triangular assembly
  still loops; batching it needs the `tri` geometry setup made
  trace-safe first (Phase 2 work).
- [x] Deliverable: a `geodef[jax]` extra and `benchmarks/bench_greens.py`
  (caching disabled, sequential runs) comparing NumPy vs JAX assembly for a
  range of patch/observation counts. Measured on a plain multi-core CPU:
  10-50x steady-state speedup for rectangular displacement assembly, with
  JIT compilation (~1.5-3 s) paid once per problem shape.

- [x] Vectorized DC3D rewrite (`okada92`): the scalar Fortran-style port
  (per-point control flow, module-level common blocks) was rewritten as
  pure, vectorized, `where`-based array math routed through the backend
  namespace, keeping line-by-line formula correspondence with the
  published DC3D.f. Anchored to golden data generated from the scalar
  port (shear), validated by finite-difference displacement gradients
  (all components), and equivalence-tested on JAX. The rewrite also
  restored the tensile-fault DU entries that the scalar port had
  truncated (opening-mode strains previously omitted the z-derivative
  and part of the y-derivative terms). `greens.strain_greens` and the
  `okada` dispatcher now evaluate all observation points per patch in
  one call — the 100-patch fault self-stress kernel assembles ~11x
  faster on NumPy, and the kernel is now available to the JAX backend
  for the stress-shadows / earthquake-cycle path.

### Phase 2 — Differentiable forward model (IN PROGRESS)
- [x] `geodef.gradients`: `rect_displacement(theta, slip, ...)` traceable in
  the native rectangle parameters and `tri_displacement(vertices, slip, ...)`
  traceable in the vertex coordinates, plus `*_jacobian` helpers built on
  `jax.jacfwd` returning `(d, ∂d/∂θ, ∂d/∂m)`. All Jacobians validated
  against central finite differences (`tests/test_gradients.py`); the `tri`
  path needed setupTDCS's construction asserts skipped under tracing and
  the image-triangle mirror rewritten without `np.copy`/in-place writes.
- [x] Full `G(θ)` assembly: `gradients.rect_greens(theta, ...)` builds the
  (3*nobs, 2*N) displacement Green's matrix for a discretized planar
  fault via a traced mirror of `Fault.planar` (same patch ordering and
  layout as `greens.displacement_greens`, validated against that
  pipeline); `gradients.tri_greens(vertices, ...)` does the same for a
  triangular mesh (differentiable eagerly; jit/vmap over the mesh axis
  still open). `gradients.los_project` maps displacement G onto InSAR
  look vectors, matching `InSAR.project`. G(θ) Jacobians validated
  against finite differences.
- [ ] Remaining Phase 2: differentiable strain/stress kernels, and
  jit/vmap support for the triangular mesh axis (needs the remaining
  geometry-scalar branches in `tri.py` expressed as `where`).
- **Differentiation variables (settled 2026-07).** The `tri` engine
  differentiates with respect to the **vertex coordinates** — its native
  parameterization, well-defined for any mesh including non-planar ones.
  Gradients in terms of derived parameters (trace position, dip, depth of
  a planar mesh) come free via the chain rule through a small
  `θ → vertices` builder traced by JAX. The rectangular engine
  differentiates its own native parameters (position, strike, dip,
  length, width) directly.
- Validate gradients against finite differences on small problems in tests;
  take care near the `tri` angular-dislocation branch boundaries where the
  `where`-selected configuration switches.
- Deliverable: `geodef.gradients` (or similar) giving `∂d/∂θ` and `∂d/∂m` for a
  fault + dataset.

### Phase 3 — Accelerated and gradient-based inversion (IN PROGRESS)
- [x] Batched ABIC sweep: `abic_curve` evaluates all lambdas in one batched
  JAX computation (broadcast solves + batched slogdet), transparently when
  the JAX backend is active. ABIC first per user priority; L-curve and CV
  sweeps can follow the same pattern.
- [x] `invert.geometry_search`: gradient-based nonlinear planar-fault
  geometry inversion (variable projection, wls inner solve, L-BFGS-B on
  exact forward-mode gradients, per-parameter bounds, Gauss-Newton
  covariance). One module-level jitted kernel returns residual + Jacobian,
  so multi-start calls share the compilation. Raises unless the JAX
  backend is active.
- [x] Tutorial 11 (`11_gradient_geometry.ipynb`): end-to-end gradient-based
  geometry inversion on the notebook-10 scenario — single-parameter parity,
  joint dip+depth recovery with error bars, practical notes on
  non-convexity, lambda selection, and float32 exploration. Executed by
  pytest, skipped without JAX; tutorials 01-10 stay on the NumPy path.
- [x] Opt-in float32 mode validated end-to-end (kernels, assembly, sweep)
  with the explore-in-float32 / finalize-in-float64 workflow documented.
- [ ] Batched L-curve and cross-validation sweeps on JAX (same pattern as
  the ABIC sweep).
- [ ] Later Bayesian path: HMC/NUTS over the differentiable geometry
  objective (ties to roadmap item 4). Triangular (vertex-parameterized)
  geometry optimization — e.g. along-strike dip variation on a large
  strike-slip fault — is deliberately a separate, higher-dimensional step.

### Risks and non-goals
- Do **not** make JAX a hard dependency or complicate the base API.
- Watch float32/float64 accuracy on GPU; the kernels are sensitive near the
  fault. Default to float64 and document any precision trade-offs.
- Keep the teaching notebooks on the NumPy path so they stay portable.

---

## 2. Quasi-dynamic earthquake-cycle modeling

Extend from static dislocations to time-dependent slip. Port the quasi-dynamic
rate-and-state machinery from `related/stress-shadows/unicycle/` into an optional
`geodef.cycle` module: stress-kernel-driven slip evolution on the existing
triangular/rectangular meshes, an ODE integrator, and rate-and-state friction.
This is a large phase; stage it as its own multi-step plan. The v1.0 stress
kernels and mesh I/O are the natural foundation.

---

## 3. Additional Green's-function engines

Broaden the physics behind `G` while keeping the unified dispatcher interface:

- **Meade (2007)** triangular dislocations (an alternative to Nikkhoo & Walter
  for cross-checking).
- **Compound dislocation model (CDM)** for volcanic/point sources.
- **Layered half-space** Green's functions (e.g. via a propagator-matrix or
  wavenumber-integration backend) for depth-varying elastic structure.

Each new engine should cross-validate against a published reference and slot in
behind the existing `okada`-style dispatcher so `Fault`/`greens`/`invert` gain it
for free.

---

## 4. Bayesian nonlinear inversion (example-level)

A worked MCMC study (`emcee`, already used in a `shakeout_v2` notebook) for
posterior uncertainty on fault geometry — the natural sequel to tutorial 10's
outlook. Home it in `examples/` rather than the teaching path, and let it share
the differentiable forward model from item 1 once that lands (gradient-based
samplers such as HMC/NUTS become attractive there).

---

## Design notes (unchanged, carried forward)

- Public coordinates use local Cartesian `x=East`, `y=North`, `z=Up` plus
  geographic lat/lon where appropriate. Internal Green's-function conventions are
  converted at module boundaries.
- Slip columns are blocked as `[:N]` strike-slip and `[N:]` dip-slip.
- Hash-based caching handles reuse of expensive Green's and regularization
  matrices.
- Optional dependencies stay optional: `geo` (pyproj), `mesh` (meshpy),
  `maps` (cartopy), and any future `gpu`/`cycle` extras must not burden the base
  install or the tutorials.

## Maintenance policies

- When executing a step listed in this plan, update `PLAN.md` in the same logical
  unit so the roadmap remains current.
- Keep docs `.md` files up to date alongside minor code changes. Large docs or
  typing rewrites should be their own plan step.
- Do not add `Co-Authored-By` trailers to commit messages. AI co-authorship is
  tracked once in `README.md`; update that model list plus `CLAUDE.md` and
  `AGENTS.md` when a new AI model materially contributes.
