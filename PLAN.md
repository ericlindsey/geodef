# PLAN.md — Forward-Looking Roadmap for GeoDef

**GeoDef v1.1 has shipped.** The runtime library, the eleven-part tutorial
course, the per-module documentation, and clean `ruff`/`mypy`/warning-free
tooling are all complete, and the headline JAX accelerator (roadmap item 1) has
substantially landed: the differentiable forward models, gradient-based
`geometry_search`, and the collapsed Bayesian sampler `geodef.bayes` (Phase 4)
are all in. This document is **forward-looking**: the checklists below record the
completed JAX phases for context, but the open work is the unchecked items
(batched L-curve/CV sweeps, differentiable strain/stress and triangular-mesh
jit/vmap) plus roadmap items 2–4.

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
- Remaining Phase 2, split into two independent tracks:
  - **Tri jit/vmap over the mesh axis** (unblocks 6c). `gradients.tri_greens`
    still loops over triangles because a few per-triangle,
    vertex-dependent Python branches in `tri.py` break tracing. The
    remaining blockers, with `TDdispHS` as the target entry point:
    - [x] `build_tri_coordinate_system`: the horizontal-element strike
      degeneracy (`if norm(Vstrike)==0` plus the image-dislocation
      `if tri[0][2]>0` flip) rewritten as `where`, selecting the
      Northward/Southward replacement *before* `normalize` so the 0/0
      never reaches the divide. Validated numpy-identical and
      jit-traceable on generic, horizontal-subsurface, and
      horizontal-above-surface triangles.
    - [ ] `AngSetupFSC`: the vertical-TD-side degeneracy
      (`if abs(beta)<eps or abs(pi-beta)<eps` -> zeros) as `where`; the
      delicate part is guarding `ey1 = normalize([SideVec_x, SideVec_y,
      0])` so a vertical side does not nan-poison gradients in the
      non-selected branch (same 0/0 discipline as above and the okada
      arctan fix).
    - [ ] `TDdispHS`: the two data-dependent `assert all(...<=0)` skipped
      under tracing (as `setupTDCS`'s asserts already are), and the two
      surface-triangle branches (`if all(tri[:,2]==0)` -> flip vertical
      component, negate) as `where` on the scalar `xp.all(tri[:,2]==0)`.
    - [ ] `gradients.tri_greens`: replace the Python triangle loop with a
      `jit`ed `vmap` over the mesh axis once the above land; validate the
      vmapped `G(vertices)` and its `jacfwd` against the current eager
      path and the Matlab references, and update the docstring/PLAN.
  - [ ] **Differentiable strain/stress kernels** (earthquake-cycle path,
    roadmap item 2 — *not* required for 6c): make `TDstrainHS` / the DC3D
    strain kernel traceable and gradient-safe the same way.
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
- [ ] Bayesian path: now planned concretely as Phase 4 below (supersedes the
  earlier "later Bayesian path" note; ties to roadmap item 4).

### Phase 4 — Bayesian solver (`geodef.bayes`) — PLANNED 2026-07

**Goal.** Full posterior inference for fault geometry and hyperparameters on
CPU, using the differentiable pipeline built in Phases 1–3. The design is a
**collapsed (Rao–Blackwellized) sampler**: because slip is linear given
geometry, the (up to hundreds of) slip parameters are marginalized
analytically — the same Cholesky/log-determinant math the batched ABIC sweep
already uses (ABIC is −2·log marginal likelihood up to priors, Fukuda &
Johnson 2008) — and NUTS samples only the ~5–10 dimensional space of geometry
plus noise/regularization scales. Per-sample cost is a few ms after one JIT
compile, so full 4-chain runs finish in minutes on a plain CPU. Slip
uncertainty (now including geometry uncertainty) comes afterward from the
exact Gaussian conditional p(slip | theta, lambda, sigma, d), vmapped over
posterior draws.

Decisions:
- **Sampler library: blackjax** (JAX-native, lightweight, functional) behind
  a new optional extra `geodef[bayes]` (= jax + blackjax). No pyro/numpyro.
  `emcee` stays out of the package — used only in the validation example.
- **Sampled parameters:** free geometry components of theta (as in
  `geometry_search`), plus `log_sigma` (noise scale) and `log_lambda`
  (regularization scale). Priors: uniform/normal per geometry parameter,
  wide log-uniform defaults on the scales.
- **Rectangular engine first.** Triangular-mesh geometry sampling waits for
  the Phase 2 tri jit/vmap work.

Steps (red/green TDD, one commit per step):
- [x] Reverse-mode safety of the rect path. Investigation overturned the
  expected failure mode: the `cos(dip)` `where` branches were already
  reverse-mode safe, and the real hazard was Okada's
  `arctan(xi*eta/(q*R))` term, whose autodiff produced `0*inf = nan` in
  **both** modes when an observation lies exactly on the fault-plane ray
  (`q == 0`, routinely hit by symmetric synthetic grids). Fixed via the
  reciprocal identity `arctan(u) = sign(u)*pi/2 - arctan(1/u)` with
  well-conditioned branch selection; primal values preserved (Matlab
  reference tests) and `jax.grad` validated against `jacfwd` and finite
  differences. Remaining documented caveat: dip gradients lose accuracy
  within ~0.01 deg of exactly vertical (cancellation inherent in the
  published `1/cos(dip)` formulas, both AD modes equally).
- [x] `geodef.bayes.RectPosterior`: collapsed log-posterior over free
  geometry + `log10_sigma` (+ `log10_lambda`), slip marginalized via
  Cholesky. Three prior modes: `hierarchical` (sampled lambda),
  `weak` (identity prior, fixed slip scale — the collapsed analog of
  "unsmoothed" MCMC), and `profiled` (fixed lambda, no Occam terms).
  Uniform-prior parameters are bound-clipped before the kernels so
  gradients stay finite at rejected points. Validated exactly against a
  dense multivariate-normal reference (matrix determinant lemma),
  against `_abic_value` on an injected identical linear system, and
  against finite-difference gradients.
- [x] Sampler driver: `bayes.sample(post, ...)` wrapping blackjax NUTS with
  window adaptation behind a new `geodef[bayes]` extra; all chains drawn
  in one jitted `vmap` computation; `PosteriorResult` dataclass with
  split R-hat and Geyer/Stan effective sample size (in-house, no arviz),
  `summary()`, and a corner-style `plot_pairs()`. Chains start from the
  warmup end position overdispersed by the adapted posterior scale
  (restarting at `x0` forced every chain to re-walk the approach to the
  mode at the small adapted step size); explicit `inits` supported for
  multimodality probes.
- [x] Conditional slip posterior: `slip_mode(x)`, `slip_draws(samples)`
  (one exact Gaussian conditional draw per posterior sample — the
  Rao-Blackwell completion, so per-patch statistics include geometry and
  hyperparameter uncertainty), and `predict(samples)` for data-space
  posterior predictive fields. Draw moments validated against the
  analytic conditional Gaussian.
- [x] Validation + docs: `emcee` cross-check on the same jitted logpdf in
  the test suite (`tests/test_bayes.py`) and in the worked example
  `examples/bayesian_geometry.ipynb` (roadmap item 4's example), which
  also compares Gauss-Newton error bars from `geometry_search` against
  the full posterior and the weak-prior mode; `docs/bayes.md` added and
  module tables updated. Implementation note discovered en route: XLA
  compilation of the reverse-mode gradient through the nested okada85
  subfunctions took minutes, so `logpdf` carries a forward-mode
  `custom_jvp` rule (`jax.jacfwd`; linear in tangents, so `jax.grad`
  transposes through it exactly) — right-sized for the few sampled
  parameters and compiling in seconds.
- [ ] Step 6 — beyond the collapsed sampler (planned 2026-07; sub-steps
  below, one commit each, in order).

#### Phase 4, step 6 — positivity, triangular geometry, tempering

**Key finding on positivity vs. marginalization.** The analytic collapse
exists only because the Gaussian slip prior is conjugate; truncating it to
the positive orthant turns the marginal likelihood into a ratio of orthant
probabilities with no closed form at hundreds of dimensions. So positivity
puts slip back into the sampled state — but the speed is recovered another
way: whitened joint sampling has a per-leapfrog gradient cost of one
`(3*nobs, 2N)` matvec, jit/vmapped over chains, and that same compiled code
is the GPU path. The dimension cost lands in NUTS trajectory length, which
whitening keeps in check. Where a component's prior stays Gaussian, it can
still be marginalized exactly (half-collapse). Precedent for joint
slip + hyperparameter sampling under positivity: Fukuda & Johnson (2008);
we add gradients.

- [x] **6a. `bayes.SlipPosterior` — joint slip sampling with positivity,
  fixed geometry.** Sampled state = slip (as whitened `z`, one per
  component) + `log10_sigma` (+ `log10_lambda` in hierarchical mode);
  per-component positivity masks (e.g. dip-slip only) applied by softplus
  after a whitened affine map — `z` pushed through the Cholesky factor of a
  fixed reference system `H0 = Gᵀ_w G_w + lambda_ref LᵀL`, centered at the
  reference ridge solution `mu0`, with the map's log-Jacobian carried in
  the density so the posterior is the exact truncated-Gaussian-prior
  posterior. Geometry is fixed, so G assembles once and `fault` may be any
  `Fault` (rectangular or triangular mesh); each gradient is one matvec, so
  plain reverse-mode `jax.grad` suffices (no `custom_jvp`). Modes:
  `hierarchical` / `fixed` / `weak` (no `profiled` — nothing is profiled
  when slip is sampled). **Correction to the earlier plan note:**
  hierarchical lambda is *not* biased under positivity. The truncated prior
  is zero-mean with covariance ∝ `(sigma²/lambda)(LᵀL)⁺`, and an orthant is
  a cone, so rescaling the covariance moves no mass across its boundary —
  the normalizer Z is the *same constant* for every `(sigma, lambda)` and
  cancels. Sampled lambda is therefore exact as built (would only bias for a
  nonzero prior mean, e.g. a `smoothing_target`). Validated: `logpdf` vs an
  independent NumPy reference; the exact "joint = collapsed × Gaussian
  conditional" identity against `RectPosterior`; end-to-end sampler
  agreement with the collapsed posterior; positivity posterior mean vs
  `LinearSystem.invert(bounds=(0, None))`; `emcee` cross-check; gradients vs
  finite differences. (`tests/test_bayes_slip.py`, `docs/bayes.md`.)
- **6b. Joint geometry + slip with positivity, then half-collapse.**
  Staged in two commits (decided 2026-07).
  - [x] **6b-1. Joint geometry + full slip sampling.** `RectPosterior`
    gains a `positive=` argument; `positive=None` stays exactly the
    collapsed sampler (unchanged code path), while setting it makes the
    slip prior truncated so the whole slip vector rejoins the sampled
    state as a whitened block appended after the hyperparameters and is
    sampled jointly with geometry. Reuses 6a's whitened-softplus transform
    (extracted to the shared `_slip_transform` helper) with free `theta`
    through `rect_greens`. Key implementation point: the differentiation
    `custom_jvp` is placed around `G(theta)` **alone** (its tangent is
    `jacfwd` over the 7 geometry params), so plain reverse-mode `jax.grad`
    over the whole `logpdf` traces the Okada kernel only 7 times per
    gradient regardless of the (large) slip block — unlike the collapsed
    path's whole-`logpdf` forward-mode wrapper, which would scale with the
    slip dimension. Validated: exact "joint = collapsed × Gaussian
    conditional" identity (all-False mask, ~1e-11), gradient vs finite
    differences through the kernel, geometry+positive-slip recovery, and
    an `emcee` cross-check. (`tests/test_bayes_slip.py`, `docs/bayes.md`.)
  - [x] **6b-2. Half-collapse (efficiency).** Marginalizes the
    *unconstrained* slip block analytically (Gaussian conditional given
    the constrained block) so only geometry + hyperparameters + the
    sign-constrained components are sampled — halves the slip dimension in
    the common one-constrained-component (`positive='dip'`,
    `components='both'`) case. Made **automatic**: `positive` alone
    decides which components stay in the state; the rest are always
    marginalized. Closed form: with `H_f = Gᵀ_f G_f + λ K_ff`,
    `b = Gᵀ_f r_c − λ K_cfᵀ m_c`,
    `S_c = ‖r_c‖² + λ m_cᵀ K_cc m_c − bᵀ H_f⁻¹ b`, the log-marginal is
    `−(n+p_c)/2·log(2πσ²) + ½(r logλ + logdet_sum) − ½logdet H_f
    − S_c/(2σ²)` [+ orthant], reducing to the collapsed formula at
    `p_c=0` and to 6b-1 at `p_f=0`. The constrained block is whitened by
    the **Schur complement** of the reference `H0` (its exact marginal
    precision, = `H0` when nothing is marginalized), so `_slip_transform`
    is reused unchanged. `slip_draws` completes the marginalized block
    from its exact Gaussian conditional per draw (seed-dependent), while
    the constrained block stays deterministic and non-negative.
    Validated: the marginal vs an independent NumPy reference of the
    `H_f`/`S_c` formula (~1e-11), exact reduction to the collapsed
    posterior at `p_c=0`, gradients vs finite differences through the
    marginalization, a partial bool mask, and the split determinism of
    `slip_draws`.
- [ ] **6c. Triangular-mesh geometry sampling (`bayes.TriPosterior`).**
  Prerequisite: the remaining Phase 2 tri jit/vmap work. Parameterization
  decision: **no remeshing inside the sampler** — connectivity flips are
  posterior discontinuities and break static shapes. Instead, freeze one
  reference mesh (connectivity plus per-vertex (along-strike, down-dip)
  parameter coordinates) and let theta be values at a coarse set of
  control knots (corners plus a few interior knots for curvature — depth
  or local dip at a small (u, v) grid, ~5–15 parameters), pushed through a
  fixed smooth interpolant (tensor-product spline or RBF with fixed
  centers) that warps every vertex. Differentiable, one jit, chain rule
  through the traced theta -> vertices builder per the Phase 2 decision.
  Bounds come from knot priors (dip in (0, 90), seismogenic depth range)
  with softplus increments where monotonicity matters. The slip prior
  stays Gaussian here, so the full collapse still applies. Tests must
  cover warps crossing the angular-dislocation branch boundaries.
- [ ] **6d. Tempered SMC (`bayes.sample_smc`).** Glue around blackjax
  `adaptive_tempered_smc`: prior-draw sampler from the parsed prior
  specs, the existing NUTS kernel as the mutation step, adaptive
  schedule; returns a `PosteriorResult` plus a log-evidence estimate
  (free model comparison). De-risks 6a/6b multimodality. Independently
  deferrable if blackjax API churn makes it costly.

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
outlook. Home it in `examples/` rather than the teaching path. With item 1's
Phase 4 (`geodef.bayes`) now planned, this example doubles as its validation:
run `emcee` on the same posterior the NUTS sampler targets and confirm the
two agree on a small problem.

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
