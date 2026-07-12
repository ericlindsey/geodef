# Changelog

All notable changes to GeoDef are recorded here, newest first. The format
follows [Keep a Changelog](https://keepachangelog.com/); versions follow the
compatibility policy in [COMPATIBILITY.md](COMPATIBILITY.md). Entries that
change numerical output beyond documented tolerances are tagged **numerical**.

## [Unreleased]

### Added

- `geodef.medium.ElasticMedium`: one declared home for half-space elastic
  parameters. `Fault` accepts `medium=` in every factory (and
  `fault.with_medium(...)` copies a fault with new parameters); Poisson's
  ratio now actually reaches the dislocation kernels from the high-level
  path (it was previously frozen at 0.25 below `Fault.greens_matrix`), and
  `moment`/`magnitude`/`stress_kernel` default their `mu` to the fault's
  medium. Green's and stress cache keys include the medium.
- MIT `LICENSE`, `CITATION.cff` citation metadata, and complete package
  metadata (authors, URLs, classifiers, dependency minimums).
- `py.typed` marker: downstream type checkers now see GeoDef's annotations.
- GitHub Actions CI: lint/typecheck, base-install and all-extras test tiers
  on Python 3.10/3.13 (Linux and macOS), and an oldest-supported-dependency
  job (numpy 1.24, scipy 1.10, matplotlib 3.7).
- Documentation consistency tests: public members must be documented, doc
  examples must parse, documented names must exist.
- README capability table mapping features to modules and install extras.

### Fixed

- **numerical** — ABIC (all three implementations: `compute_abic`,
  `LinearSystem` sweeps, and the batched JAX sweep) filtered prior
  eigenvalues with a plain `> 0` test, so a Laplacian's numerically-zero
  modes (~1e-15) leaked a spurious `n0 * log(lambda)` term into the
  criterion and biased automatic smoothing selection. The same filter
  appeared in the Bayesian posteriors' prior pseudo-determinants
  (`RectPosterior`, `TriPosterior`, `SlipPosterior`), biasing sampled
  smoothing strengths identically. Eigenvalues are now cut at a
  numerical-rank threshold (as in `matrix_rank`) everywhere. A new
  convention test pins the scaling invariance this restores.
- The regularized objective was written as `lambda^2 ||L m||^2` in
  tutorials 04/09/10 and the course outline while every solver uses
  `lambda ||L m||^2`; the text now matches the code, and
  `docs/conventions.md` records the mapping to published `alpha`/
  `lambda^2` notations.

### Changed

- **breaking** — `GNSS`, `InSAR`, `Vertical`, `Fault.planar`, and
  `Fault.planar_from_corner` are now keyword-only (and `from_triangles`
  after `vertices`): the geographic ordering policy is lon-first for new
  named APIs, and positional lat/lon pairs could be silently swapped.
  `Fault.centers_geo` provides `[lon, lat, depth]` centroids matching
  `Mesh.centers_geo`; `Fault.centers` keeps its legacy latitude-first
  order and is documented as such.
- **breaking** — `InversionResult.chi2` is renamed `reduced_chi2` (it held
  the reduced statistic while `DatasetDiagnostics.chi2` held the unreduced
  one). Accessing `.chi2` now raises an `AttributeError` with migration
  guidance instead of silently changing meaning; `.npz` files written
  before the rename still load. Vocabulary: `chi2` = unreduced `r^T W r`,
  `reduced_chi2` = `chi2 / dof` (docs/conventions.md).
- **numerical** — `model_covariance` / `model_uncertainty` now return the
  linear-Gaussian posterior covariance `H^-1` by default (`H = GtWG +
  lambda LtL`), matching what Tutorial 09 teaches and what `geodef.bayes`
  samples; the previous frequentist estimator covariance
  `H^-1 GtWG H^-1` remains available as `kind='estimator'`. The two agree
  when unregularized.
- Every module reference page links `docs/conventions.md`, the single
  reference for axes, depth sign, angles, units, array ordering, and the
  regularization/misfit conventions (enforced by a docs test).
- Disk-cache keys now include the implicit compute context: a
  `cache.KERNEL_VERSION` stamp plus the active backend and precision, so
  kernel fixes and float32/float64 or NumPy/JAX switches can never serve a
  stale cached matrix. Existing cache entries are orphaned (recomputed on
  first use); run `geodef.cache.clear()` to reclaim the disk space.
- The `mesh` extra now installs `netCDF4` (required by `Mesh.from_slab2`).
- The package version is single-sourced from `geodef.__version__`.
- Mesh-generation tests skip (rather than fail) when `meshpy`/`netCDF4` are
  not installed, matching every other optional-dependency test.
- Golden okada92 comparisons use `rtol=1e-9` so platform/NumPy roundoff
  drift does not produce false failures.

## [1.1.0] - 2026-07

Baseline for this changelog; see `git log` for prior detail.

- Rectangular (Okada 1985/1992) and triangular (Nikkhoo & Walter 2015)
  dislocation engines with surface and internal displacement/strain paths.
- `Fault`, `GNSS`, `InSAR`, `Vertical` domain objects; rectangular and
  triangular mesh construction; Green's assembly and disk caching.
- Weighted, bounded, constrained, fixed-direction, and regularized linear
  slip inversion with L-curve, ABIC, cross-validation, uncertainty,
  resolution, and per-dataset diagnostics.
- Optional JAX backend with differentiable forward models, gradient-based
  geometry search, and collapsed/positivity-aware Bayesian inference
  (NUTS via blackjax).
- Eleven-part executed tutorial course, worked examples, and per-module
  reference documentation.
