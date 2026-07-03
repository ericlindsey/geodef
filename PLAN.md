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
- [ ] `geodef.backend`: `set_backend()`/`get_backend()`, `GEODEF_BACKEND` env
  var, array-namespace resolution (NumPy default, JAX when installed and
  selected), float64 default with opt-in float32.
- [ ] Trace-safe kernel refactor (pure NumPy, behavior-preserving): rewrite
  `tri.trimodefinder`'s `flatnonzero` + fancy-index assignment into
  `where`/mask form, and make the `okada92` numerical core branch-free
  (`allow_singular` raise moves outside the core). Validate against the
  Matlab reference `.npz` files at existing tolerances.
- [ ] Route the `okada85`/`okada92`/`tri` elementwise math through the backend
  namespace; keep NumPy the default everywhere — invisible to existing users
  and the tutorials. Note: `greens.py` assembles `G` with Python loops over
  patches, so the JAX path needs a `vmap`-over-patches assembly, not just
  elementwise routing.
- [ ] Cross-validate JAX output against the existing Matlab reference `.npz`
  files to the same tolerances the CPU engines meet (tests skipped when JAX
  is absent).
- [ ] Deliverable: identical results, a `geodef[jax]` extra, and a benchmark
  harness (apples-to-apples, caching disabled) comparing NumPy vs JAX
  assembly for a range of patch/observation counts.

### Phase 2 — Differentiable forward model
- Express `G(θ)` assembly as a JAX-traceable function of the geometry parameters
  `θ` (position, strike, dip, length, width for rectangles; vertices for
  triangles) and expose `jax.jacobian`/`jax.grad` of the predicted data with
  respect to `θ` and slip, for **both** engines.
- Validate gradients against finite differences on small problems in tests;
  take care near the `tri` angular-dislocation branch boundaries where the
  `where`-selected configuration switches.
- Deliverable: `geodef.gradients` (or similar) giving `∂d/∂θ` and `∂d/∂m` for a
  fault + dataset.

### Phase 3 — Accelerated and gradient-based inversion
- JAX-backed linear solves for the regularized normal equations (the `wls`
  path); batched hyperparameter sweeps (L-curve / ABIC / CV) evaluated in
  parallel across `λ` via `vmap`. Bounded/constrained solvers remain on SciPy.
- Gradient-based nonlinear geometry search: replace the grid-then-`minimize`
  recipe of tutorial 10 with a differentiable objective and
  `scipy.optimize.minimize` (L-BFGS-B) fed by JAX gradients, with the linear
  slip solved inside (variable projection). Optional gradient-based / HMC
  sampling as a later Bayesian path.
- Deliverable: an accelerated `invert(...)` path selected by backend, plus a new
  advanced example notebook demonstrating end-to-end differentiable geometry
  inversion — and honest CPU-vs-GPU benchmarks.

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
