# `geodef.bayes` — Collapsed Bayesian geometry inference

Full posterior inference for planar-fault geometry and noise /
regularization hyperparameters, with the linear slip parameters
**marginalized analytically** (a collapsed, Rao-Blackwellized sampler).
Because slip enters the forward model linearly, its Gaussian integral has
a closed form built from the same Cholesky / log-determinant quantities
the ABIC machinery uses (ABIC is `-2 log` marginal likelihood up to
constants; Yabuki & Matsu'ura 1992, Fukuda & Johnson 2008). NUTS
therefore explores only the ~3-10 dimensional space of geometry plus
scales — minutes on a plain CPU — instead of hundreds of correlated slip
dimensions.

Requires the JAX backend and blackjax:

```python
import geodef
geodef.backend.set_backend("jax")   # pip install geodef[bayes]
```

---

## The statistical model

```
d = G(theta) m + e        e ~ N(0, sigma^2 W^-1)
m ~ N(0, sigma^2 (lambda L^T L)^-1)
```

- `theta` — planar-fault geometry `[e0, n0, depth, strike, dip, length,
  width]`; any subset can be sampled (`free`), the rest stay fixed.
- `sigma` — dimensionless noise scale factor multiplying the dataset
  covariances (`sigma = 1` means the reported data errors are exact).
  Sampled as `log10_sigma`.
- `lambda` — slip-prior (regularization) strength. In hierarchical mode
  it is sampled as `log10_lambda`, so results **average over all
  smoothing strengths weighted by the evidence** — no single lambda is
  ever chosen.

The slip `m` never appears in the sampled space: given `(theta, sigma,
lambda)` its conditional posterior is exactly Gaussian and is recovered
after sampling (see "Slip posterior" below).

## Slip-prior modes

| mode | prior on slip | sampled scales | use it for |
|------|--------------|----------------|------------|
| `'hierarchical'` | smoothing operator `L` (e.g. Laplacian), `lambda` sampled | `log10_sigma`, `log10_lambda` | the default: hierarchical Bayes over smoothness |
| `'weak'` | `L = I`, fixed `slip_scale` | `log10_sigma` | the collapsed analog of "unsmoothed" MCMC: resolved patches tighten, unresolved patches stay honestly wide |
| `'profiled'` | fixed `lambda`, **no Occam terms** | `log10_sigma` | comparison with `geometry_search`; not a proper marginal posterior |

---

## Building a posterior

```python
post = geodef.bayes.RectPosterior(
    theta0,                       # [e0, n0, depth, strike, dip, length, width]
    [gnss, insar],                # any DataSet mix
    ref_lat=-2.0, ref_lon=100.0,  # local Cartesian anchor
    free=["dip", "depth"],
    theta_prior={
        "dip": (5.0, 60.0),               # uniform
        "depth": ("normal", 25e3, 5e3),   # or normal
    },
    n_length=8, n_width=4,
    components="both",            # slip columns to marginalize
    mode="hierarchical",
    smoothing="laplacian",
    smoothing_strength=1.0,       # initial lambda for the sampler
)

post.param_names   # ['dip', 'depth', 'log10_sigma', 'log10_lambda']
post.x0            # starting point (theta0 values + scale defaults)
post.logpdf(x)     # traceable, differentiable log-posterior
```

`logpdf`, `log_likelihood`, and `log_prior` are pure JAX-traceable
functions of the sampled vector `x` — hand them to any JAX sampler or
optimizer. Uniform-prior parameters are clipped to their bounds before
entering the elastic kernels, so `jax.grad` stays finite even at
rejected points.

Start `theta0` from your best estimate — e.g. a `geometry_search`
result — since warmup adapts around it.

## Sampling

```python
result = geodef.bayes.sample(
    post, n_samples=2000, n_warmup=1000, n_chains=4, seed=0
)

result.samples          # (n_chains, n_samples, n_params)
result.flat             # (n_chains*n_samples, n_params)
result.rhat, result.ess # split R-hat and effective sample size per parameter
result.n_divergent      # should be ~0
result.summary()        # dict: mean, sd, q05, q50, q95, rhat, ess
result.plot_pairs(truths=[15.0, 25e3, 0.0, 1.0])   # corner plot
```

`sample` runs blackjax window adaptation once (step size + diagonal mass
matrix), then draws all chains in a single jitted `vmap` computation.
Chains start from the warmup's end position, overdispersed by twice the
adapted posterior scale; pass `inits=(n_chains, n_params)` explicitly to
probe multimodality from dispersed starts.

Convergence checklist: `rhat < 1.01`-ish, `ess` in the hundreds,
`n_divergent == 0`. The standalone diagnostics are exported as
`bayes.split_rhat(chains)` and `bayes.effective_sample_size(chains)`.

## Slip posterior and predictions

```python
thin = result.flat[::10]                  # thin if you like
draws = post.slip_draws(thin, seed=1)     # (n, n_slip) joint posterior slip
mean_slip = draws.mean(axis=0)
ci90 = np.percentile(draws, [5, 95], axis=0)

pred = post.predict(thin, seed=2)         # (n, n_data) noise-free predictions
```

`slip_draws` completes the collapsed sampler: one exact Gaussian
conditional draw per posterior sample yields draws from the **joint**
posterior `p(m, theta, scales | d)`, so per-patch statistics include
geometry and hyperparameter uncertainty — something no fixed-geometry
linear inversion can provide. `slip_mode(x)` returns the conditional
mode (the regularized least-squares slip) at a single sample.

Column layout matches the inversion convention: `[:N]` strike-slip,
`[N:]` dip-slip when `components='both'`.

---

## Positivity and joint slip sampling: `SlipPosterior`

`RectPosterior` marginalizes slip analytically — which is possible *only*
because the slip prior is Gaussian. The moment you want a **non-Gaussian
slip prior** — most commonly a **positivity constraint** (dip-slip that
cannot go negative on a thrust, or a locking fraction confined to
`[0, 1]`) — the Gaussian integral no longer has a closed form, and slip
has to go back into the sampled state.

`SlipPosterior` does exactly that at **fixed geometry**: it samples the
slip vector jointly with `log10_sigma` (and, in hierarchical mode,
`log10_lambda`). Because the geometry is fixed, the Green's matrix is
assembled **once** at construction and never re-touched, so `fault` may be
*any* `Fault` — rectangular **or a triangular mesh** — and each `logpdf`
gradient costs a single matrix-vector product (plain reverse-mode
`jax.grad`, no forward-mode wrapper).

```python
post = geodef.bayes.SlipPosterior(
    fault,                       # any Fault, fixed geometry
    [gnss, insar],
    components="dip",
    mode="fixed",                # a proper posterior at one lambda
    smoothing="laplacian",
    smoothing_strength=1.0,
    positive="dip",              # dip-slip constrained >= 0
)

result = geodef.bayes.sample(post, n_samples=2000, n_warmup=1000, n_chains=4)
draws = post.slip_draws(result.flat)   # (n, n_slip), every row >= 0 where constrained
pred = post.predict(result.flat)       # (n, n_data) posterior-predictive data
```

**Positivity is exact, not a penalty.** A whitened, softplus
reparameterization enforces `m >= 0` on the constrained components
identically: a fixed reference system `H0 = Gᵀ_w G_w + lambda_ref LᵀL`
(Cholesky factor `L0`) defines an affine map `z -> v` centered at the
reference ridge solution, then constrained components pass through
`softplus(v)` and unconstrained ones through unchanged. The map's
log-Jacobian is carried in the density, so the posterior is the true
truncated-Gaussian-prior posterior, not a soft barrier.

- `positive` selects the constrained components: `None`, `'strike'`,
  `'dip'`, `'both'` (all sampled components), or a bool array of length
  `n_slip`.
- `slip_draws(samples)` is a **deterministic transform** here (no `seed`):
  slip is part of the sampled state, so each posterior sample already *is*
  one slip vector. `slip_of(x)` maps a single sample.

### Mode names differ on purpose

| mode | prior on slip | sampled scales |
|------|--------------|----------------|
| `'hierarchical'` | smoothing operator `L`, `lambda` sampled | `log10_sigma`, `log10_lambda` |
| `'fixed'` | smoothing operator `L`, `lambda` fixed — a **proper** posterior at one lambda (Occam terms included) | `log10_sigma` |
| `'weak'` | `L = I`, fixed `slip_scale` | `log10_sigma` |

There is **no `'profiled'` mode**: nothing is profiled out here (slip is
sampled, never point-estimated), so `RectPosterior`'s `'profiled'` — which
drops the Occam log-determinant terms — has no analog. `'fixed'` is the
honest single-lambda posterior.

### Hierarchical lambda stays exact under positivity

The truncated slip prior is a zero-mean Gaussian restricted to an orthant,
with covariance proportional to `(sigma²/lambda)·(LᵀL)⁺`. Its truncation
normalizer `Z` — the probability mass inside the orthant — is what would,
in general, depend on the hyperparameters and bias a sampled `lambda`.
But an orthant is a **cone**, and rescaling a zero-mean Gaussian's
covariance by a scalar moves no mass across a cone boundary: `Z` is the
*same constant* for every `(sigma, lambda)` and cancels out of the
posterior. So `mode='hierarchical'` remains exact with positivity — no
correction needed. (This would break for a **nonzero** prior mean, e.g. a
`smoothing_target`, where the orthant is no longer a cone about the mean.)

`SlipPosterior` shares `sample()`, the diagnostics, and `predict` with
`RectPosterior`; only the geometry-vs-slip split of the sampled vector
differs. Validation: the density is checked against an independent NumPy
reimplementation and against `RectPosterior`'s collapsed formula via the
exact "joint = collapsed × Gaussian conditional" identity, the sampler
against the collapsed posterior and against `emcee`, and the constrained
posterior mean against `LinearSystem.invert(bounds=(0, None))`.

### Positivity *with* free geometry: `RectPosterior(positive=…)`

`SlipPosterior` holds geometry fixed. When you want positivity **and** an
uncertain geometry — sample the two jointly — pass `positive` straight to
`RectPosterior`:

```python
post = geodef.bayes.RectPosterior(
    theta0, [gnss], ref_lat=-2.0, ref_lon=100.0,
    free=["dip", "depth"],                 # geometry sampled...
    theta_prior={"dip": (5.0, 45.0), "depth": (10e3, 40e3)},
    n_length=8, n_width=4,
    components="both", mode="hierarchical", smoothing="laplacian",
    positive="dip",                        # ...and dip-slip constrained >= 0
)
result = geodef.bayes.sample(post)
draws = post.slip_draws(result.flat)       # deterministic here; >= 0 where constrained
```

With `positive=None` (the default) `RectPosterior` is exactly the
collapsed sampler documented above — nothing changes. Setting `positive`
makes the slip prior truncated on the selected components, which is no
longer conjugate *there*.

**Only the constrained components rejoin the sampled state** — the
unconstrained slip is still marginalized analytically (a *half-collapse*).
So `positive='dip'` with `components='both'` samples the dip block and
integrates the strike block out, and `param_names` becomes `[*free,
log10_sigma, (log10_lambda), z…]` with one `z` per **constrained**
component. The sampled dimension therefore grows by `p_c` (constrained
count), not by the full slip size. Two limits fall out for free:
`positive` constraining nothing reduces exactly to the collapsed sampler,
and constraining everything gives a fully joint slip sampler.

The constrained block is whitened by the Schur complement of a reference
system built once at `theta0` (its exact marginal precision), and `G(θ)`
is re-assembled per evaluation — but only the assembly sits inside a
`custom_jvp`, so each gradient traces the Okada kernel **seven times**
(once per geometry parameter) no matter how many components are sampled.
The marginalization adds one `p_f × p_f` Cholesky per evaluation.

`slip_draws` returns the **full** slip: the constrained block is the
deterministic softplus map (non-negative exactly, `seed`-independent),
while the marginalized block is completed from its Gaussian conditional —
a genuine draw, so it uses `seed`. `slip_mode` sets that block to its
conditional mean instead. The half-collapse marginal is validated against
an independent NumPy reference of the `H_f`/`S_c` formula, against the
collapsed posterior in the all-marginalized limit, and by an `emcee`
cross-check.

Cost note: even half-collapsed, positivity sampling is markedly heavier
than the collapsed sampler (tens–hundreds of dimensions instead of a
handful), so prefer collapsed `RectPosterior`/`SlipPosterior` unless you
specifically need positivity or another non-Gaussian slip prior *together
with* geometry uncertainty.

---

## Practical notes

- **Cost.** One `logpdf` gradient evaluation is a Green's assembly plus
  one small Cholesky — milliseconds after the one-time JIT compile.
  A 4-chain run on a tutorial-scale problem takes minutes on CPU.
- **Accuracy near vertical faults.** Geometry gradients lose accuracy
  within ~0.01 degrees of `dip = 90` (cancellation in the published
  Okada `1/cos(dip)` terms, both AD modes). Keep dip bounds off exactly
  90 if that sliver matters; values (and hence the accept/reject
  decision) remain accurate.
- **Rank-deficient smoothing.** For a Laplacian with a null space the
  marginal uses the pseudo-determinant convention, matching
  `LinearSystem._abic_value`; the data must constrain the null-space
  directions.
- **ABIC connection.** Maximizing the hierarchical marginal over
  `sigma` at fixed `lambda` reproduces the ABIC objective up to a
  constant — `abic_curve` is the point-estimate shortcut of this
  posterior.
- **Validation.** The marginal is tested exactly against a dense
  multivariate-normal density (matrix determinant lemma), against the
  ABIC machinery on an identical linear system, and against `emcee` on
  the same posterior (`examples/`); gradients against finite
  differences.
