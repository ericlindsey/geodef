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
