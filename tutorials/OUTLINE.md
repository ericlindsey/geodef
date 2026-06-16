# GeoDef Tutorial Notebooks — Master Outline

This document is the **design plan for the tutorial notebook series**. It is the
single source of truth for what each notebook teaches, the order concepts are
introduced, the math that must appear, and the (deliberately small) code each
notebook uses. Write or revise notebooks against this outline; revise this
outline first if the plan changes.

---

## 1. Purpose and Audience

These notebooks are a **course in geodetic inverse methods**, taught through the
GeoDef API — not an API tour. A reader who finishes the series should understand
how surface geodetic data (GNSS, InSAR) are turned into images of fault slip,
*why* each step is done, and what the results can and cannot be trusted to say.
GeoDef is the vehicle; the destination is the methods.

Concretely, every notebook is organized around **concepts and equations first,
code second**:

- **Theory is primary.** Each notebook develops the relevant math (forward
  operator, least squares, regularization, resolution, …) with equations and
  short conceptual prose. A student should be able to read the markdown alone
  and learn the method.
- **Code is minimal and illustrative.** Snippets are short — usually a few lines
  — and exist to *demonstrate* the concept just explained, not to build a
  production workflow. Favor one clear `geodef.*` call plus a labeled plot over
  long scripts.
- **Data is synthetic and reproducible.** Tutorials generate their own simple
  geometries and noise with fixed random seeds so they run fast, execute under
  pytest, and isolate the concept being taught.
- **Plots are labeled and inline.** Visualization is woven throughout (see
  §4) rather than taught as a separate topic.
- **Each notebook ends with exercises** that push the reader to vary a parameter
  and predict/observe the effect.

### Tutorials vs. Examples

| | `tutorials/` | `examples/` |
|---|---|---|
| Goal | Teach one method per notebook | Show a complete real workflow |
| Data | Synthetic, tiny, seeded | Real, bundled or downloaded |
| Code | Short snippets | Full end-to-end scripts |
| Math | Developed in detail | Assumed; referenced |
| Tested | Executed by `tests/test_tutorials.py` | Smoke-tested at most |

Longer worked examples (e.g. the Gorkha earthquake) stay under `examples/`.
When a tutorial would balloon into a full case study, the case study moves to
`examples/` and the tutorial keeps only the minimal teaching version.

---

## 2. Disposition of the Existing Notebooks (01–04)

The current `01_forward_model`, `02_caching`, `03_plotting`, and
`04_mesh_generation` notebooks date to an earlier, API-introduction framing.
They are **retired in their current form** and their content is redistributed:

| Old notebook | Disposition |
|---|---|
| `01_forward_model` | Rebuilt as the new **Tutorial 01** with dislocation theory added; the joint GNSS+InSAR section moves forward to **Tutorial 06**. |
| `02_caching` | Demoted to a short **sidebar/appendix in Tutorial 02** ("computing G is expensive; GeoDef caches it"). Not a standalone tutorial. |
| `03_plotting` | Dissolved. Each plot type is introduced in the tutorial where its concept first appears (see §4). The exhaustive gallery survives as a non-tutorial reference notebook or doc, not a numbered tutorial. |
| `04_mesh_generation` | Moved out of the teaching sequence. Basic rectangular vs. triangular discretization is covered conceptually in **Tutorial 02**; real mesh building (traces, polygons, slab2.0) becomes an **`examples/` worked example**, since it is data/IO-heavy rather than a method. |

Net effect: the numbered tutorials become the 10-part methods sequence below,
and utility material (caching, plotting gallery, mesh IO) is either absorbed as
supporting detail or relocated to `examples/`.

---

## 3. Shared Conventions

So notebooks stay consistent and composable:

- **Coordinates.** Local Cartesian `x = East`, `y = North`, `z = Up`; geographic
  `lat`/`lon` where geometry is built. State units on every quantity (m, km, °).
- **Slip vector blocking.** `m = [s_strike(0..N-1), s_dip(0..N-1)]`, i.e. the
  first `N` entries are strike-slip, the next `N` are dip-slip.
- **Forward operator.** Always written `d = G m` (data = Green's matrix × slip).
- **Notation (used across all notebooks).**
  - `d` — data vector, `m` — model (slip) vector, `G` — Green's/design matrix.
  - `C_d` — data covariance, `C_m` — posterior model covariance.
  - `W = C_d^{-1}` — data weight matrix; `λ` — regularization weight.
  - `L` — regularization operator (Laplacian, identity, stress kernel).
  - `N` — number of patches; `2N` — number of slip parameters (both components).
  - `R` — model resolution matrix.
- **Recurring synthetic scenario.** A single small fault is reused across
  notebooks so readers track one problem end to end. Default teaching geometry:
  a planar fault via `Fault.planar(...)` discretized coarsely (e.g. `n_length`
  ≈ 8, `n_width` ≈ 5) so `G` is small and inversions are instant. A prescribed
  "true" slip patch (a smooth bump) generates synthetic data; seeded Gaussian
  noise is added. Notebooks reuse this setup via a short copied cell rather than
  a shared import, so each notebook stays self-contained.
- **Random seeds.** Every notebook sets `rng = np.random.default_rng(0)` (or a
  stated seed) for reproducible noise and pytest stability.
- **Standard imports.** `import numpy as np`, `import matplotlib.pyplot as plt`,
  `import geodef`.

---

## 4. Visualization Strategy

There is no standalone plotting tutorial. Each `geodef.plot` function is
introduced at the moment its underlying concept is taught:

| Plot | Introduced in | Because |
|---|---|---|
| `plot.slip`, `plot.patches` | 01 | first time slip-on-fault is shown |
| `plot.map`, `plot.vectors` | 01 | surface displacement field |
| `plot.fault3d` | 02 | discretized geometry in 3D |
| `plot.fit` | 03 | observed vs. predicted diagnostic |
| `plot.insar` | 06 | first InSAR dataset |
| L-curve / ABIC curve plots | 05 | hyperparameter selection |
| `plot.resolution`, `plot.uncertainty` | 09 | model assessment |

The retired plotting gallery may live on as `tutorials/reference_plots.ipynb`
(not numbered, not part of the methods path) or fold into `docs/plot.md`.

---

## 5. The Ten Tutorials

Each entry below specifies: **Goal**, **Concepts & Math** (what equations/theory
must appear), **Key API** (the small code surface used), **Plots**, and
**Exercises**. Notebooks should follow this structure literally.

---

### Tutorial 01 — Forward Model Basics
*The elastic dislocation forward problem.*

**Goal.** Understand how slip on a buried fault produces surface displacement,
and compute it for a single rectangular source.

**Concepts & Math.**
- Faulting as a *dislocation* in an elastic half-space; homogeneous, isotropic,
  linear elasticity.
- The representation theorem in words: surface displacement is an integral of
  slip over the fault weighted by the medium's elastic Green's functions.
- The Okada (1985) rectangular dislocation as the analytic building block;
  inputs are geometry (location, `strike`, `dip`, `length`, `width`, `depth`)
  and slip (`strike-slip`, `dip-slip`, optional opening).
- Geometry conventions: strike, dip, rake; decomposition of a slip vector of
  given rake into strike-slip and dip-slip components.
- The map: `u(x) = G(x; geometry) · s` — for *fixed* geometry the displacement
  is **linear in slip**. (This linearity is the hinge for Tutorials 02–09.)
- Near-field vs. far-field: how displacement amplitude decays with distance and
  source depth.

**Key API.** `Fault.planar(...)`, `fault.displacement(...)` (or
`greens.greens` + a slip vector), basic `Fault` attributes (`n_patches`,
`patch_outlines`).

**Plots.** Map-view surface displacement (`plot.map` / `plot.vectors`); slip on
the fault patch (`plot.slip`).

**Exercises.** Vary dip and depth; predict then observe how the surface pattern
and peak amplitude change. Flip the rake (thrust → normal) and interpret the
vertical field.

---

### Tutorial 02 — Fault Discretization and the G Matrix
*From a continuous fault to a linear system.*

**Goal.** Build the Green's matrix for a multi-patch fault and understand its
structure as the discrete forward operator.

**Concepts & Math.**
- Discretizing a fault surface into `N` patches; slip approximated as
  piecewise-constant per patch.
- Superposition (from linearity): total displacement is the sum of each patch's
  unit-slip response ⇒ the discrete forward problem `d = G m`.
- Anatomy of `G`: column `j` is the surface response to **unit slip on patch
  `j`**; rows are observation components. Block layout: first `N` columns
  strike-slip, next `N` columns dip-slip. Units of `G` entries.
- How dataset projection turns the raw 3-component response into observed rows
  (interleaved E/N/U for GNSS; foreshadow LOS for InSAR in Tut 06).
- Rectangular vs. triangular patches (conceptual): why triangular meshes fit
  curved/variable-dip geometry; pointer to the `examples/` mesh workflow.
- **Sidebar (absorbs old caching notebook):** `G` (and stress kernels) can be
  expensive; GeoDef hashes inputs and caches results to disk, so repeated calls
  are instant. One short demo, then move on.

**Key API.** `Fault.planar(...)` with multiple patches, `greens.greens(fault,
dataset)`, inspecting `G.shape`; brief `geodef.cache.info()` sidebar;
`plot.fault3d`.

**Plots.** 3D fault geometry colored by patch index/depth; a single column of
`G` rendered as the surface response of one patch.

**Exercises.** Refine `n_length`/`n_width` and watch `G.shape` and column count
grow; plot the response of an edge patch vs. a central patch.

---

### Tutorial 03 — Unregularized Inversion
*Least squares, and why raw inversion fails.*

**Goal.** Estimate slip from data by (weighted) least squares, and see ill-posed
inversion overfit noise.

**Concepts & Math.**
- The linear inverse problem `d = G m + ε`, noise `ε` with covariance `C_d`.
- Ordinary least squares: minimize `‖G m − d‖²`; the normal equations.
- **Weighted** least squares with `W = C_d^{-1}`:
  `m̂ = (Gᵀ W G)^{-1} Gᵀ W d`. Why weighting by data uncertainty matters.
- Over- vs. under-determined systems; rank deficiency and the role of the
  condition number of `Gᵀ W G`.
- Goodness of fit: residuals, weighted misfit `χ²`, reduced `χ²`.
- **Overfitting demo:** invert noisy synthetic data with no regularization and
  watch the recovered slip oscillate wildly while fitting the data "too well."
  Motivates Tutorial 04.

**Key API.** `geodef.invert(fault, dataset)` (default WLS),
`InversionResult` (`.slip_vector`, `.predicted`, misfit fields), `plot.fit`,
`plot.slip`.

**Plots.** True vs. recovered slip side by side; observed-vs-predicted scatter
(`plot.fit`); the noisy/oscillatory unregularized solution.

**Exercises.** Increase the noise level and re-invert; reduce the number of
stations below the number of patches and observe instability.

---

### Tutorial 04 — Regularization
*Making the inverse problem well-posed.*

**Goal.** Stabilize the inversion with prior information and understand the
common regularization operators.

**Concepts & Math.**
- Ill-posedness ⇒ need for prior constraints (Tikhonov regularization).
- The regularized objective:
  `min_m ‖G m − d‖²_{C_d} + λ² ‖L (m − m_ref)‖²`.
- Choices of `L`:
  - **Smoothing** — discrete Laplacian; penalizes slip *roughness*.
  - **Damping** — `L = I`; penalizes slip *magnitude* (minimum norm / moment).
  - **Stress-kernel** — physically motivated operator from inter-patch stress
    interactions.
  - Role of `m_ref` (regularize toward a reference model, default zero).
- The augmented/stacked linear system view: regularization = adding synthetic
  "equations" with weight `λ`.
- Qualitative effect of `λ`: under- vs. over-smoothing (sets up Tutorial 05).

**Key API.** `geodef.invert(..., smoothing='laplacian'|'damping'|'stresskernel',
smoothing_strength=λ, smoothing_target=m_ref)`; `greens` Laplacian builder
referenced conceptually.

**Plots.** A small panel grid: recovered slip at several `λ` values from
under- to over-smoothed.

**Exercises.** Swap Laplacian for damping and compare; sweep `λ` by eye and
guess a good value (then check it in Tutorial 05).

---

### Tutorial 05 — Choosing Regularization Strength
*Principled selection of λ.*

**Goal.** Replace eyeballing `λ` with quantitative criteria.

**Concepts & Math.**
- The misfit–roughness trade-off as `λ` varies.
- **L-curve:** plot model norm vs. data misfit (log–log); the corner balances
  the two. How the corner is located.
- **ABIC** (Akaike Bayesian Information Criterion): the Bayesian/marginal-
  likelihood view of regularization as a prior, with `λ` a hyperparameter;
  minimize ABIC.
- **Cross-validation:** hold out data, predict it, pick `λ` minimizing
  prediction error; `k`-fold mechanics.
- When the methods agree/disagree and how to choose between them.

**Key API.** `geodef.lcurve(...)`, `geodef.abic_curve(...)`,
`geodef.compute_abic(...)`, and `geodef.invert(..., smoothing_strength='abic'`
`|'cv', cv_folds=...)`; their built-in curve plots.

**Plots.** L-curve with marked corner; ABIC vs. `λ`; CV error vs. `λ`; the
chosen solution.

**Exercises.** Compare the `λ` chosen by L-curve, ABIC, and CV on the same
problem; change noise level and see which criterion is most stable.

---

### Tutorial 06 — Multiple Datasets
*Joint inversion of GNSS and InSAR.*

**Goal.** Combine complementary datasets in one inversion and handle their
relative weighting.

**Concepts & Math.**
- Why joint: GNSS (sparse, 3-component, absolute) vs. InSAR (dense, 1-D
  line-of-sight, relative). Complementary spatial sampling.
- **InSAR line-of-sight projection:** scalar LOS = `u · l̂`, the look-vector
  geometry (`look_e`, `look_n`, `look_u`); ascending vs. descending.
- Stacking the forward problem: vertically concatenate `G` and `d`; block data
  covariance `C_d`.
- **Relative weighting** between datasets and its effect on the solution;
  connection to the regularization hyperparameter (foreshadow multi-λ).

**Key API.** `geodef.GNSS(...)`, `geodef.InSAR(...)` with look vectors,
`geodef.invert(fault, [gnss, insar], ...)`, `plot.insar`, `plot.vectors`.

**Plots.** GNSS vectors and InSAR LOS for the same scenario; joint-inversion
slip vs. single-dataset slip; per-dataset fit panels.

**Exercises.** Down-weight one dataset and watch the slip migrate toward the
other; add a second (descending) InSAR track.

---

### Tutorial 07 — Correlated Noise and InSAR
*Beyond diagonal data covariance.*

**Goal.** Represent and exploit spatially correlated noise, especially in
InSAR.

**Concepts & Math.**
- Why InSAR noise is **spatially correlated** (atmosphere, orbits) — the
  diagonal-`C_d` assumption of earlier notebooks is wrong here.
- Building a full covariance matrix from a covariance function
  (exponential/Gaussian) with a **correlation length** and variance.
- Effect of the off-diagonal terms: data "whitening", effective number of
  independent observations, and the impact on both the slip estimate and its
  uncertainty.
- Practical down-sampling of dense InSAR as a related concern (brief).

**Key API.** `geodef.InSAR(...)` with a full covariance / covariance-function
specification; `geodef.invert(...)` with the resulting `C_d`.

**Plots.** A covariance matrix / covariance function; inversion with diagonal
vs. full `C_d` and the difference in recovered slip and uncertainty.

**Exercises.** Vary the correlation length from ~0 (white) to large; observe how
the solution and its error bars respond.

---

### Tutorial 08 — Bounds and Constraints
*Enforcing physically admissible slip.*

**Goal.** Add inequality and sign constraints, and reduce parameters by fixing
slip direction.

**Concepts & Math.**
- Why constraints: slip should not reverse sense; magnitude bounds; geologic
  priors. These cannot be expressed by quadratic regularization alone.
- **Non-negative least squares (NNLS):** `m ≥ 0`.
- **Bounded least squares:** general `lb ≤ m ≤ ub`.
- **Linear inequality constraints:** `C m ≤ d` as a quadratic program (QP).
- **Fixed-direction bases** (dimensionality reduction): solve one amplitude per
  patch at a fixed `rake` (`components='rake'`) or fixed geographic `azimuth`
  (`components='azimuth'`), which also encodes a sign/sense prior cleanly.
- Trade-offs: constraints vs. smoothing; bias vs. admissibility.

**Key API.** `geodef.invert(..., bounds=(0, None))` (auto-NNLS),
`bounds=(lb, ub)` (bounded LS), `method='constrained', constraints=(C, d)`,
`components='rake', rake=...`, `components='azimuth', slip_azimuth=...`.

**Plots.** Unconstrained vs. non-negative slip (spurious back-slip removed);
fixed-rake one-component slip vectors.

**Exercises.** Compare WLS, NNLS, and a fixed-rake inversion on the same data;
add an upper bound and find where it begins to bias the fit.

---

### Tutorial 09 — Uncertainty and Assessment
*How well is the slip actually resolved?*

**Goal.** Quantify and visualize model uncertainty and resolution, and report
derived quantities.

**Concepts & Math.**
- **Posterior model covariance:**
  `C_m = (Gᵀ W G + λ² LᵀL)^{-1}` (linear-Gaussian result); diagonal ⇒
  per-patch slip uncertainty.
- **Model resolution matrix** `R = G^{-g} G` (generalized inverse times `G`):
  each recovered patch is a *weighted average* of the truth; rows of `R` are
  resolution kernels.
- The **resolution–uncertainty trade-off** as a function of regularization.
- **Checkerboard / restitution tests:** recover a known synthetic pattern to
  map where the data resolve slip.
- Derived quantities: scalar **moment** and **moment magnitude** from the slip
  estimate, with uncertainty.

**Key API.** `geodef.model_covariance(...)`, `geodef.model_resolution(...)`,
`geodef.model_uncertainty(...)`, `geodef.dataset_diagnostics(...)`,
`fault.moment(...)` / magnitude; `plot.resolution`, `plot.uncertainty`.

**Plots.** Per-patch uncertainty map; resolution diagonal map; a checkerboard
input/recovered pair.

**Exercises.** Run a checkerboard test at two `λ` values and relate the
recoverable checker size to the resolution map; report `M_w` ± uncertainty.

---

### Tutorial 10 — Nonlinear Geometry Search
*When the geometry itself is unknown.*

**Goal.** Estimate nonlinear fault parameters (location, strike, dip, depth)
on top of the linear slip inversion.

**Concepts & Math.**
- Why geometry is **nonlinear**: `G` depends nonlinearly on fault position and
  orientation, so `d = G(θ) m` is not linear in `θ`.
- **Separable (variable-projection) structure:** for any trial geometry `θ`,
  slip `m` is still a *linear* inversion (Tutorials 03–08); the outer search is
  over the few nonlinear parameters only.
- Objective surface over `θ`; local minima; the value of a coarse **grid
  search** before gradient/quasi-Newton refinement.
- Optimization with `scipy.optimize`: defining the misfit-as-function-of-θ,
  starting points, bounds.
- **Outlook (no heavy code):** fully Bayesian sampling of `θ` with MCMC
  (`emcee`) for posterior uncertainty on geometry — pointer to a future
  `examples/` study.

**Key API.** A small Python objective wrapping `geodef.invert(...)` inside
`scipy.optimize.minimize` / a grid loop; reuse of earlier inversion calls.

**Plots.** Misfit vs. a scanned geometry parameter (e.g. dip); recovered vs.
true geometry and slip.

**Exercises.** Grid-search dip, then refine with `minimize`; perturb the
starting geometry to find a local minimum and discuss mitigation.

---

## 6. Authoring Checklist (per notebook)

Before a notebook is considered done:

- [ ] Opens with a title + goal + bullet list of what the reader will learn.
- [ ] Develops the math/theory in markdown **before** the code that uses it.
- [ ] Reuses the shared conventions and (where relevant) the recurring synthetic
      scenario from §3.
- [ ] Keeps code cells short and focused; no hidden complexity.
- [ ] Sets a fixed random seed; runs top-to-bottom under
      `tests/test_tutorials.py` in seconds.
- [ ] All plots are labeled (titles, axis labels, units, colorbars).
- [ ] Ends with 2–4 exercises that vary a parameter and ask for a prediction.
- [ ] `tutorials/README.md` updated to list the notebook.

---

## 7. Open Design Questions

Resolve these as the series is built; record decisions here.

1. **Shared scenario delivery.** Copy the setup cell into each notebook (max
   self-containment) vs. a tiny `tutorials/_scenario.py` helper (less
   duplication). Current lean: copy the cell, keep notebooks standalone.
2. **Plotting gallery.** Keep the old exhaustive gallery as an unnumbered
   `reference_plots.ipynb`, or migrate its content into `docs/plot.md`?
3. **Mesh generation home.** Confirm the trace/polygon/slab2.0 workflow lands as
   an `examples/` study rather than a tutorial.
4. **Tutorial 07 covariance API.** Confirm the exact `InSAR` covariance-function
   entry point before writing, and align with `docs/data.md`.
5. **Numbering during transition.** Whether to renumber files immediately or
   stage the new notebooks alongside the old ones until the series is complete.
