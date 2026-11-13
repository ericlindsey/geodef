# Changelog

All notable changes to GeoDef are recorded here, newest first. The format
follows [Keep a Changelog](https://keepachangelog.com/). GeoDef is pre-1.0:
the API may change between releases, and this file records how. Entries that
change numerical output beyond documented tolerances are tagged **numerical**.

## [Unreleased]

### Fixed

- **numerical** Rectangles and triangles are flattened by one map
  projection. Rectangular patches were each placed in their own tangent
  plane while observation points and triangles used the fault's shared
  frame, so a large fault was not one plane: subdividing a 300 km fault at
  40° latitude into 5 x 4 patches changed its uniform-slip displacement by
  0.3% (1.4 mm per meter of slip), and the JAX geometry model behind
  `geometry_search` and `RectPosterior` disagreed with `Fault.planar` by 1%
  for that fault. Every patch, vertex, and observation now goes through
  `fault.frame`: subdivision is exact to 1e-13, the JAX model matches
  `Fault.planar` to 1e-12, and a rectangle and the two triangles that tile
  it agree to rounding. `LocalFrame` now keeps depth as
  height rather than measuring it along the curved Earth, and `to_enu` /
  `to_geographic` are exact inverses. For faults a few tens of kilometers
  across the change is below 1e-7 relative. `fault.strike` is now the azimuth in
  the fault's frame; `Fault.geographic_strike` gives it from true north, and
  fault files and slip-azimuth and plate-motion bases use the true-north
  value. `KERNEL_VERSION` is bumped.
- **`Fault.planar` assigned row depths in reverse.** With `n_width > 1`
  the row on the up-dip side of the plane was given the deepest depth and
  the row on the down-dip side the shallowest, so a subdivided planar
  fault was not the plane its arguments described: uniform slip on a
  2-row fault differed from the single-patch solution by tens of percent.
  The same sign error was copied into the JAX layout used by
  `gradients.rect_greens`, `invert.geometry_search`, and
  `bayes.RectPosterior`, and into `planar_from_corner`. Row 0 is now the
  shallowest, as documented, and subdividing a fault no longer changes
  its uniform-slip displacement. Single-row faults, file loaders, and
  triangular meshes were unaffected. Any saved result, figure, or
  geometry estimate from a multi-row `Fault.planar` should be
  regenerated. `KERNEL_VERSION` is bumped so cached matrices are rebuilt.

### Changed — fewer surprises

- The disk cache for Green's matrices and stress kernels is **off by
  default**. GeoDef no longer creates `.geodef_cache/` in the working
  directory unless `geodef.cache.enable()` is called.

### Changed — one way to do each thing

A simplification pass removed parallel spellings. There are no aliases;
update calls as follows.

| Before | After |
|---|---|
| `data.gnss(...)`, `data.horizontal_gnss(...)`, `GNSS(ve=, vn=, vu=, se=, sn=, su=)` | `GNSS(lon=, lat=, east=, north=, up=, sigma=)`; omit `up` for horizontal-only; `sigma_east`/`sigma_north`/`sigma_up` override `sigma` |
| `data.insar(...)`, `data.vertical(...)` | `InSAR(...)`, `Vertical(...)` (scalars broadcast) |
| `GNSS(name=<station labels>, dataset_name=...)` | `GNSS(station_names=..., name=...)`; `dataset.name` is the dataset name |
| `data.from_table(table, kind="vertical", columns=...)` | `Vertical.from_table(table, columns=...)` |
| building synthetic data from `G @ m` and `[0::3]` slices | `GNSS.synthetic(fault, lon=, lat=, dip_slip=, sigma=, rng=)` |
| `fault.displacement(lat, lon, slip_strike=, slip_dip=)` | `fault.displacement(lon=, lat=, strike_slip=, dip_slip=)`; keeps the shape of `lon`/`lat` |
| `fault.greens_matrix(lat, lon, obs_depth=)` | `fault.greens_matrix(lon=, lat=, depth=)` |
| `fault.vertices_2d`, `vertices_3d` (depth in km), `patch_outlines` | `fault.corners` (local meters), `fault.corners_geo` (`[lon, lat, depth_m]`), for both rectangles and triangles |
| `Fault(lat, lon, depth, ..., engine=...)` | `Fault(lon=, lat=, depth=, ...)`, keyword-only; the engine follows from `vertices` |
| `invert.model_covariance(result, fault, data)`, `model_uncertainty`, `model_resolution` | `result.covariance()`, `result.uncertainty()`, `result.resolution()` |
| `invert.prediction(result)`, `residual`, `diagnostics`, `summary` | `result.prediction()`, `.residual()`, `.diagnostics()`, `.summary()` |
| `invert.prediction(result)[gnss.dataset_name]` | `result.prediction(gnss)` (works for unnamed datasets) |
| `invert.save(result, path)`, `invert.load(path)`, `invert.save_table(result, path, fault)` | `result.save(path)`, `InversionResult.load(path, fault=, datasets=)`, `result.save_table(path)` |
| `invert.lcurve(fault, data, ...)`, `invert.abic_curve(fault, data, ...)` | `system = geodef.LinearSystem(fault, data, regularization=...)`; `system.lcurve(...)`, `system.abic_curve(...)` |
| `LinearSystem.invert(...)` | `LinearSystem.solve(...)` |
| `geodef.okada85`, `geodef.okada92`, `geodef.tri`, `geodef.okada` | `geodef.kernels.okada85`, `.okada92`, `.tri`, `.okada` |
| `greens.displacement_greens(lat, lon, lat0, lon0, depth, strike, dip, L, W)` and the strain / `tri_` variants | keyword-only local Cartesian: `displacement_greens(obs_east=, obs_north=, east=, north=, depth=, strike=, dip=, length=, width=)`; `tri_displacement_greens(obs_east=, obs_north=, vertices=)` |
| `LocalFrame(..., projection="wgs84-enu")` | `LocalFrame(lon=, lat=, alt=)`; there is one projection |
| `LocalFrame(lat, lon)`, `frame.origin_lat` | `LocalFrame(lon=, lat=)` (keyword-only), `frame.lat`; defined in `geodef.transforms` |
| `ref_lat=, ref_lon=` on `Fault.from_triangles`, `Fault.load`/`save`, `geometry_search`, `RectPosterior`, `Mesh.vertices_enu` | `frame=geodef.LocalFrame(lon=, lat=)` |

Also removed: the private engine registry (`_engines`), the `format=`
argument of dataset `save` (only `"dat"` existed), the dataframe-interchange
reader behind `from_table` (tables are read with `table[column]`), and the
API-tier stability map, layering test, and deprecation policy that enforced
the previous structure. `geodef.LinearSystem` is exported at the top level.

### Changed

- Renamed the regularization keyword family for a single, math-aligned
  vocabulary: `smoothing` → `regularization`, `smoothing_strength` →
  `regularization_strength`, and `smoothing_target` → `regularization_target`
  across `invert`, `bayes`, and the `InversionResult` record. The operator
  values `'laplacian'`, `'damping'`, and `'stresskernel'` are unchanged.
- Renamed `greens.resolution` to `greens.resolution_matrix` and `plot.map` to
  `plot.map_view` to remove name collisions and builtin shadowing.
- `invert.solve` now takes only `fault` and `datasets` positionally; all other
  arguments are keyword-only.
- Every public `mesh` length/depth argument is now in **meters**. `from_slab2`
  previously took `target_length` and `max_depth` in kilometers (default
  `target_length=50000.0` replaces `50.0`); `from_trace`'s `max_depth` is now
  meters, matching its meters-based `dip` callback.
- The result-file schema is now version 3; version-2 archives (which used the
  `smoothing*` field names) are migrated automatically on load.
- `geodef.__all__` is trimmed to the beginner-public vocabulary. Expert names
  (`lcurve`, `LinearSystem`, `model_*`, `stack_*`, `spatial_covariance`, the
  moment conversions, and result types) remain importable as `geodef.<name>`
  but are excluded from `__all__` and should be reached through their module.

### Removed

- The legacy latitude-first `Fault.centers` property; use `Fault.centers_geo`
  (`[lon, lat, depth]`) or `Fault.centers_local` (ENU meters).

## [0.1.0] - 2026-07-15

### Added

- Function-oriented everyday APIs: `data.gnss`, `data.horizontal_gnss`,
  `data.insar`, `data.vertical`, dataframe-neutral `data.from_table`, named
  `greens` and `slip` conversions, and explicit patch reshape/flatten helpers.
- Stable dataset identifiers, station labels, displacement/velocity semantics,
  units, epoch, and time-span metadata. Joint inversion results retain named
  row partitions, solver/regularization/backend provenance, interpretation
  warnings, reproducibility inputs, and solve-time per-dataset diagnostics
  without retaining live fault or dataset objects.
- `invert.prediction`, `invert.residual`, `invert.diagnostics`, and
  `invert.summary`, with corresponding result-aware plotting functions that do
  not require manual row slicing.
- Versioned, checksum-verified result archives through `invert.save` and
  `invert.load`: numeric-only `.npz` data, an embedded manifest, a readable
  JSON sidecar, and in-memory migration of older unversioned archives.
- `geodef.validation`: fail-early physical input validation on every public
  constructor (non-finite values, bad latitudes/dips/depths, non-positive
  uncertainties, non-unit InSAR look vectors with a `normalize_look`
  option, asymmetric/non-PD covariances with a documented escape hatch),
  plus interactive `Fault.validate()`, `DataSet.validate()`, and
  `Mesh.validate()` reports for suspicious-but-legal setups (above-surface
  patches, degenerate/sliver triangles, duplicated stations,
  satellite-to-ground look-vector sign).
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

- The constrained least-squares solver now normalizes its objective before
  calling SLSQP and rejects unsuccessful or infeasible optimizer results;
  previously, objective scaling on some supported SciPy versions could return
  a solution just outside the declared inequality constraint.
- The geodetic fixed-point iteration in `transforms.ecef2geod` looped
  forever when fed NaN coordinates (pegging a core at 100%); it is now
  iteration-bounded so non-finite inputs propagate as NaN, and constructor
  validation rejects them before they reach any transform.
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

- **breaking** — `geodef.invert` is consistently the module,
  `geodef.invert.solve(...)` is the primary inversion call, and
  `geodef.solve(...)` is the short alias. The colliding top-level callable and
  the redundant module-level `dataset_diagnostics(result, fault, datasets)`
  path were removed during the pre-release window.
- **breaking** — result I/O is functional: use `invert.save`, `invert.load`,
  and `invert.save_table`; `InversionResult` remains a compact data record
  rather than becoming a workflow facade.
- **breaking** — `GNSS`, `InSAR`, `Vertical`, `Fault.planar`, and
  `Fault.planar_from_corner` are now keyword-only (and `from_triangles`
  after `vertices`): the geographic ordering policy is lon-first for new
  named APIs, and positional lat/lon pairs could be silently swapped.
  `Fault.centers_geo` provides `[lon, lat, depth]` centroids matching
  `Mesh.centers_geo`; `Fault.centers` keeps its legacy latitude-first
  order and is documented as such.
- **breaking** — `InversionResult.chi2` is renamed `reduced_chi2` (it held
  the reduced statistic while `DatasetDiagnostics.chi2` held the unreduced
  one). This is a direct pre-release rename with no compatibility alias;
  vocabulary is now `chi2` = unreduced `r^T W r` and `reduced_chi2` =
  `chi2 / dof` (docs/conventions.md).
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

## [0.1.0] - 2026-07-12

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
