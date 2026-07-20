# API Stability Map

This page is the authoritative map of GeoDef's public API tiers (roadmap
3.1). Every public name is listed here with its tier and home module.
`tests/test_public_api.py` parses this page and fails when the code and the
map disagree, so the map cannot silently drift.

The compatibility and deprecation rules that these tiers promise are
defined in [COMPATIBILITY.md](../COMPATIBILITY.md); changes are recorded in
[CHANGELOG.md](../CHANGELOG.md). Conventions for coordinates, units, and
ordering are in [conventions.md](conventions.md).

## The three tiers

- **Beginner-public** — the small top-level vocabulary in
  `geodef.__all__`: domain objects, the one-shot `solve`, its result
  record, and the submodules that are the discovery surface for everything
  else. These names are used bare (`geodef.Fault`, `geodef.solve`) and are
  the most stable surface in the package.
- **Expert-public** — stable API reached through its module path
  (`geodef.invert.lcurve`, `geodef.greens.matrix`, `geodef.slip.pack`).
  The stability promise is the same as the beginner tier; the distinction
  is prominence and audience, not reliability. Several of these names
  (e.g. the `geodef.data` constructors) appear in the beginner course —
  spelled through their module, per the module-path policy.
- **Private** — underscore-prefixed names, module internals, and the
  reference-port interior described below. No stability promise; do not
  import these outside GeoDef.

Both public tiers follow the deprecation policy: removals and renames go
through a deprecation warning, a migration example, and at least one minor
release of overlap.

## Beginner-public tier (top level)

| Name | Kind | Defined in |
|---|---|---|
| `Fault` | class | `geodef.fault` |
| `GNSS` | class | `geodef.data` |
| `InSAR` | class | `geodef.data` |
| `Vertical` | class | `geodef.data` |
| `DataSet` | class | `geodef.data` |
| `LocalFrame` | class | `geodef.geometry` |
| `ElasticMedium` | class | `geodef.medium` |
| `DEFAULT_MEDIUM` | constant | `geodef.medium` |
| `solve` | function | `geodef.invert` |
| `InversionResult` | class | `geodef.invert` |

The submodules re-exported at the top level (the discovery surface):
`backend`, `bayes`, `cache`, `data`, `euler`, `geomap`, `geometry`,
`gradients`, `greens`, `invert`, `medium`, `mesh`, `okada`, `okada85`,
`okada92`, `plot`, `slip`, `transforms`, `tri`, `validation`.

## Expert-public tier (by module)

Every public function and class defined in each module is listed;
`tests/test_public_api.py` enforces completeness, so adding a public name
without adding it here fails CI.

### `geodef.backend`

| Name | Kind | Summary |
|---|---|---|
| `default_dtype` | function | Return the default floating-point dtype for the active precision |
| `get_backend` | function | Return the name of the active backend |
| `get_precision` | function | Return the active floating-point precision |
| `masked_eval` | function | Evaluate a vectorized function only where a mask is True |
| `namespace` | function | Return the array namespace of the active backend |
| `set_backend` | function | Select the array backend used by the compute kernels |
| `set_precision` | function | Set the floating-point precision for backend computations |
| `to_numpy` | function | Convert a backend array to NumPy at a module boundary |

### `geodef.bayes`

| Name | Kind | Summary |
|---|---|---|
| `PosteriorResult` | class | Posterior draws and diagnostics from `sample` |
| `RectPosterior` | class | Collapsed log-posterior for planar-fault geometry and scales |
| `SlipPosterior` | class | Joint log-posterior over slip and scales at fixed geometry |
| `TriPosterior` | class | Collapsed log-posterior for a warped triangular-mesh geometry |
| `TriWarp` | class | Low-dimensional normal-offset parameterization of a tri mesh |
| `effective_sample_size` | function | Effective sample size from split chains |
| `sample` | function | Sample a posterior with NUTS (blackjax) and report diagnostics |
| `split_rhat` | function | Split-chain potential scale reduction factor (R-hat) |

### `geodef.cache`

| Name | Kind | Summary |
|---|---|---|
| `cached_compute` | function | Compute a matrix, caching the result to disk |
| `clear` | function | Remove all cached files from the cache directory |
| `compute_hash` | function | Deterministic SHA-256 digest of a dict of inputs |
| `disable` | function | Disable disk caching |
| `enable` | function | Enable disk caching (the default) |
| `get_dir` | function | Return the current cache directory |
| `info` | function | Return cache statistics |
| `is_enabled` | function | Return whether caching is currently enabled |
| `set_dir` | function | Set the cache directory |

### `geodef.data`

| Name | Kind | Summary |
|---|---|---|
| `DataSet` | class | Abstract base class for geodetic data types |
| `GNSS` | class | Three-component displacement/velocity observations |
| `InSAR` | class | Line-of-sight displacement observations |
| `Vertical` | class | Single-component vertical displacement observations |
| `from_table` | function | Build a dataset from explicitly mapped table columns |
| `gnss` | function | Build a validated three-component GNSS dataset |
| `horizontal_gnss` | function | Build a validated horizontal GNSS dataset |
| `insar` | function | Build a validated InSAR line-of-sight dataset |
| `spatial_covariance` | function | Build a spatially-correlated data covariance matrix |
| `vertical` | function | Build a validated vertical-observation dataset |

### `geodef.euler`

| Name | Kind | Summary |
|---|---|---|
| `best_fit_pole` | function | Fit the best Euler pole to horizontal GNSS velocities |
| `euler_location` | function | Convert a Cartesian rotation vector to a geodetic pole |
| `euler_rot_matrix` | function | Design matrix mapping a rotation vector to velocities |
| `euler_vector` | function | Convert a geodetic Euler pole to a Cartesian rotation vector |
| `pole_velocity` | function | Predict horizontal velocities produced by an Euler pole |
| `remove_pole` | function | Subtract an Euler-pole rotation from a velocity field |

### `geodef.fault`

| Name | Kind | Summary |
|---|---|---|
| `Fault` | class | Immutable collection of fault patches with forward modeling |
| `magnitude_to_moment` | function | Convert moment magnitude to seismic moment |
| `moment_to_magnitude` | function | Convert seismic moment to moment magnitude |

### `geodef.geomap`

| Name | Kind | Summary |
|---|---|---|
| `add_fault` | function | Overlay fault patch outlines on a geographic axes |
| `add_vectors` | function | Overlay horizontal velocity vectors on a geographic axes |
| `basemap` | function | Create a Cartopy map axes with background features |

### `geodef.geometry`

| Name | Kind | Summary |
|---|---|---|
| `LocalFrame` | class | A local East-North-Up frame tied to a geographic origin |
| `as_planar_vector` | function | Validate and return the expert planar-geometry vector |
| `planar_parameter_dict` | function | Named planar parameters from a mapping or expert vector |
| `triangle_strike_dip` | function | Right-hand-rule strike and dip from triangle vertices |
| `vertices_from_nodes` | function | Expand shared ENU nodes into per-triangle vertices |

### `geodef.gradients`

| Name | Kind | Summary |
|---|---|---|
| `los_project` | function | Project a Green's matrix onto per-point look vectors |
| `rect_displacement` | function | Traceable surface displacement for a rectangular fault |
| `rect_displacement_jacobian` | function | Forward model and Jacobians, rectangular fault |
| `rect_greens` | function | Displacement Green's matrix G(theta), planar fault |
| `tri_displacement` | function | Traceable displacement for a triangular dislocation |
| `tri_displacement_jacobian` | function | Forward model and Jacobians, triangular dislocation |
| `tri_greens` | function | Displacement Green's matrix G(vertices), triangular mesh |

### `geodef.greens`

| Name | Kind | Summary |
|---|---|---|
| `build_laplacian_2d` | function | 2-D finite-difference Laplacian for a rectangular grid |
| `build_laplacian_2d_simple` | function | Simple 2-D Laplacian with free boundaries |
| `build_laplacian_knn` | function | Distance-weighted graph Laplacian from nearest neighbors |
| `displacement_greens` | function | Displacement Green's matrix for rectangular patches |
| `laplacian` | function | The fault's patch Laplacian regularization matrix |
| `matrix` | function | Projected Green's matrix for one or more datasets |
| `project` | function | Project a raw Green's matrix through a dataset's projection |
| `resolution_matrix` | function | Resolution matrix R = pinv(G) @ G |
| `select_slip_columns` | function | Project a Green's matrix onto the requested slip basis |
| `stack_obs` | function | Concatenate observation vectors across datasets |
| `stack_weights` | function | Block-diagonal inverse-covariance weight matrix |
| `strain_greens` | function | Strain Green's matrix for rectangular patches |
| `tri_displacement_greens` | function | Displacement Green's matrix for triangular patches |
| `tri_strain_greens` | function | Strain Green's matrix for triangular patches |

### `geodef.invert`

| Name | Kind | Summary |
|---|---|---|
| `ABICCurveResult` | class | Result of an ABIC curve analysis |
| `DatasetDiagnostics` | class | Per-dataset fit diagnostics |
| `GeometrySearchResult` | class | Result of a nonlinear geometry search |
| `InversionResult` | class | Result of a fault slip inversion |
| `LCurveResult` | class | Result of an L-curve analysis |
| `LinearSystem` | class | Prepared linear system for repeated analyses |
| `abic_curve` | function | Sweep regularization strength, computing ABIC |
| `compute_abic` | function | ABIC value at one regularization strength |
| `diagnostics` | function | Stored fit diagnostics keyed by dataset name |
| `geometry_search` | function | Gradient-based nonlinear geometry inversion |
| `lcurve` | function | Sweep regularization strength, computing the L-curve |
| `load` | function | Load a versioned result archive |
| `model_covariance` | function | Model covariance matrix |
| `model_resolution` | function | Model resolution matrix |
| `model_uncertainty` | function | Per-parameter 1-sigma uncertainty |
| `prediction` | function | Split stacked predictions by dataset name |
| `residual` | function | Split stacked residuals by dataset name |
| `save` | function | Save a versioned result archive plus JSON manifest |
| `save_table` | function | Save slip as a human-readable per-patch table |
| `solve` | function | Invert geodetic data for fault slip |
| `summary` | function | Assumptions and fit statistics as plain text |

### `geodef.medium`

| Name | Kind | Summary |
|---|---|---|
| `ElasticMedium` | class | Homogeneous isotropic elastic half-space parameters |

### `geodef.mesh`

| Name | Kind | Summary |
|---|---|---|
| `Mesh` | class | Immutable triangular mesh in geographic coordinates |
| `from_points` | function | Triangular mesh from scattered 3-D points |
| `from_polygon` | function | Triangular mesh from a polygon boundary |
| `from_slab2` | function | Triangular mesh from a slab2.0 NetCDF depth grid |
| `from_trace` | function | Triangular mesh from a surface trace and dip |

### `geodef.okada`

| Name | Kind | Summary |
|---|---|---|
| `displacement` | function | Unified rectangular-dislocation displacement dispatcher |

### `geodef.plot`

| Name | Kind | Summary |
|---|---|---|
| `diagnostics` | function | Compare one stored fit diagnostic across datasets |
| `fault3d` | function | 3-D visualization of fault geometry |
| `fit` | function | Observed vs. predicted values |
| `insar` | function | InSAR LOS data as colored scatter points |
| `map_view` | function | 2-D map view of fault geometry and stations |
| `patches` | function | Arbitrary scalar quantity on fault patches |
| `prediction` | function | Observed vs. predicted for every named dataset |
| `residual` | function | Residual distributions for every named dataset |
| `resolution` | function | Resolution matrix diagonal on fault patches |
| `slip` | function | Fault slip distribution as colored patches |
| `slip_interpolated` | function | Smoothly interpolated slip field in map view |
| `summary` | function | Render the plain-text inversion summary on an axes |
| `uncertainty` | function | Model uncertainty on fault patches |
| `vectors` | function | GNSS displacement/velocity vectors |

### `geodef.slip`

| Name | Kind | Summary |
|---|---|---|
| `from_azimuth` | function | Slip along a geographic azimuth to local components |
| `from_plate` | function | Plate-coordinate slip to local strike/dip components |
| `from_rake` | function | Signed amplitudes along rake to strike/dip components |
| `magnitude` | function | Unsigned physical slip magnitude per patch |
| `pack` | function | Pack slip components into the blocked vector |
| `plate_rake_from_euler` | function | Plate direction in each patch's local rake coordinates |
| `rake` | function | Physical local rake in degrees per patch |
| `to_plate` | function | Local strike/dip slip to plate coordinates |
| `unpack` | function | Unpack a blocked two-component slip vector |

### `geodef.transforms`

| Name | Kind | Summary |
|---|---|---|
| `Ellipsoid` | class | Earth reference ellipsoid parameters |
| `ecef2enu` | function | ECEF coordinates to local ENU at an origin |
| `ecef2enu_vel` | function | ECEF offset/velocity to local ENU |
| `ecef2geod` | function | ECEF coordinates to geodetic (lat, lon, alt) |
| `enu2ecef` | function | Local ENU coordinates back to ECEF |
| `enu2ecef_sigma` | function | Local ENU covariance matrix to ECEF |
| `enu2ecef_vel` | function | Local ENU velocity back to ECEF |
| `enu2geod` | function | Local ENU coordinates back to geodetic |
| `geod2ecef` | function | Geodetic coordinates to ECEF |
| `geod2enu` | function | Geodetic coordinates to local ENU at an origin |
| `geod2spher` | function | Geodetic latitude to spherical |
| `haversine` | function | Great-circle distance between two points |
| `heading` | function | Great-circle heading between two points |
| `midpoint` | function | Great-circle midpoint between two points |
| `spher2geod` | function | Spherical latitude to geodetic |
| `translate_flat` | function | Offset geographic coordinates by local ENU meters |
| `vincenty` | function | Vincenty distance and forward/back azimuths |

### `geodef.validation`

| Name | Kind | Summary |
|---|---|---|
| `ValidationIssue` | class | One finding from an interactive `validate()` call |
| `ValidationReport` | class | Collected findings from a `validate()` call |
| `as_1d_floats` | function | Coerce to a finite 1-D float array or raise precisely |
| `check_covariance` | function | Validate covariance shape, symmetry, definiteness |
| `check_finite_scalar` | function | Require a finite scalar |
| `check_positive` | function | Require strictly positive, finite elements |
| `check_range` | function | Require elements in a closed range |

## Kernel modules and the reference-port interior

`geodef.okada85`, `geodef.okada92`, and `geodef.tri` are ports of
published reference implementations (Okada 1985, Okada 1992 / DC3D,
Nikkhoo & Walter 2015). Their **entry points** are expert-public:

| Name | Summary |
|---|---|
| `okada85.displacement` | Surface displacement, rectangular source |
| `okada85.tilt` | Surface tilts, rectangular source |
| `okada85.strain` | Surface strains, rectangular source |
| `okada92.okada92` | Internal displacement/derivatives (vectorized) |
| `okada92.DC3D` | Scalar DC3D reference interface |
| `tri.TDdispHS` | Half-space triangular dislocation displacement |
| `tri.TDdispFS` | Full-space triangular dislocation displacement |
| `tri.TDstrainHS` | Half-space triangular dislocation strain |
| `tri.TDstrainFS` | Full-space triangular dislocation strain |
| `tri.strain2stress` | Strain-to-stress conversion |

Every other name in these three modules (the Chinnery terms, `I1`–`I5`,
`ux_ss`-family integrands, the angular-dislocation helpers, …) is the
**reference-port interior**: spelled without an underscore so the code
stays line-for-line traceable to the published sources, but **private
tier** — no stability promise, and no renaming to make them conventional
(see `PYTHON.md` and PLAN.md 3.2).
