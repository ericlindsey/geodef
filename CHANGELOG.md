# Changelog

All notable changes to GeoDef are recorded here, newest first. The format
follows [Keep a Changelog](https://keepachangelog.com/); versions follow the
compatibility policy in [COMPATIBILITY.md](COMPATIBILITY.md). Entries that
change numerical output beyond documented tolerances are tagged **numerical**.

## [Unreleased]

### Added

- MIT `LICENSE`, `CITATION.cff` citation metadata, and complete package
  metadata (authors, URLs, classifiers, dependency minimums).
- `py.typed` marker: downstream type checkers now see GeoDef's annotations.
- GitHub Actions CI: lint/typecheck, base-install and all-extras test tiers
  on Python 3.10/3.13 (Linux and macOS), and an oldest-supported-dependency
  job (numpy 1.24, scipy 1.10, matplotlib 3.7).
- Documentation consistency tests: public members must be documented, doc
  examples must parse, documented names must exist.
- README capability table mapping features to modules and install extras.

### Changed

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
