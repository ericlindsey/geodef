# Examples and Notebooks Overview

Covers: `related/shakeout_v2/` (all Python modules, scripts, and Jupyter notebooks)

---

## 1. Python Modules (Libraries)

### `related/shakeout_v2/fault_model.py`
**Description:** `FaultModel` class for building, loading, and managing rectangular fault patch models. Handles patch geometry, Green's function computation, and visualization.

| Class / Function | Description |
|-----------------|-------------|
| `FaultModel` | Main class for managing fault patches |
| `.add_patch(...)` | Add a single fault patch |
| `.load_patches_topleft(...)` | Load patches from file (top-left corner format) |
| `.load_patches_center(...)` | Load patches from file (center format) |
| `.load_patches_comsol(...)` | Load patches from COMSOL format |
| `.create_planar_model(...)` | Create a planar model from top-left params |
| `.create_planar_model_centered(...)` | Create a planar model from centroid params |
| `.find_patch(lat, lon)` | Find the nearest patch to a given location |
| `.get_greens(lat, lon)` | Compute Green's function matrix at given stations |
| `.get_selfstress(...)` | Compute self-stress kernels |
| `.load_pickle(fname)` / `.save_pickle(fname)` | Serialize/deserialize model |
| `.get_patch_verts_center_3d()` / `_2d()` / `_both()` | Get patch vertices in various formats |
| `.plot_patch_outlines(ax)` | Plot fault outlines on a matplotlib axis |

**Dependencies:** matplotlib, numpy, geod_transform, okada_greens

---

### `related/shakeout_v2/slip_model.py`
**Description:** `SlipModel` class for managing slip distributions on fault models. Supports static/time-dependent slip, forward modeling, and moment/magnitude calculation.

| Class / Function | Description |
|-----------------|-------------|
| `SlipModel` | Main class for slip on a FaultModel |
| `.set_fault_model(fm)` | Associate a FaultModel |
| `.random_slip()` / `.smooth_slip()` | Generate random or smoothed slip distributions |
| `.load_slip_timedep(fname)` / `.load_slip_static(fname)` | Load slip from files |
| `.save_slip(fname)` | Save slip to file |
| `.set_bc_slip(...)` | Set boundary condition slip |
| `.scale_to_magnitude(Mw)` | Scale slip to target earthquake magnitude |
| `.get_magnitude()` / `.get_moment()` | Compute Mw and seismic moment |
| `.get_moment_from_mag(Mw)` | Reverse: moment from magnitude |
| `.slip_time(t)` | Get slip at time t (for time-dependent models) |
| `.forward_model_static(lat, lon)` | Predict displacements from static slip |
| `.forward_model_dynamic(lat, lon, t)` | Predict displacements at time t |

**Dependencies:** numpy, geod_transform, okada_greens, fault_model

---

### `related/shakeout_v2/geod_transform.py`
**Description:** Comprehensive geodetic coordinate transformations. Defines the WGS84 ellipsoid and provides conversions between geodetic, ECEF, ENU, and spherical frames.

| Function | Description |
|----------|-------------|
| `Ellipsoid` (dataclass) | Stores ellipsoid parameters (a, f, e, e2, b) |
| `WGS84` | Pre-defined WGS84 ellipsoid constant |
| `geod2ecef(lat, lon, alt)` | Geodetic to ECEF Cartesian |
| `ecef2geod(x, y, z)` | ECEF to geodetic (iterative) |
| `ecef2enu(x, y, z, lat0, lon0, alt0)` | ECEF to local East-North-Up |
| `ecef2enu_vel(vx, vy, vz, lat0, lon0)` | ECEF velocity to ENU velocity |
| `enu2ecef(e, n, u, lat0, lon0, alt0)` | ENU to ECEF |
| `enu2ecef_vel(ve, vn, vu, lat0, lon0)` | ENU velocity to ECEF velocity |
| `enu2ecef_sigma(se, sn, su, lat0, lon0)` | ENU uncertainties to ECEF |
| `geod2spher(lat, lon)` | Geodetic to geocentric spherical |
| `spher2geod(lat_c, lon)` | Geocentric spherical to geodetic |
| `geod2enu(lat, lon, alt, lat0, lon0, alt0)` | Geodetic directly to ENU |
| `enu2geod(e, n, u, lat0, lon0, alt0)` | ENU directly to geodetic |
| `translate_flat(lat, lon, de, dn)` | Flat-earth approximation translation |
| `vincenty(lat1, lon1, lat2, lon2)` | Vincenty distance on ellipsoid |
| `haversine(lat1, lon1, lat2, lon2)` | Haversine (spherical) distance |
| `heading(lat1, lon1, lat2, lon2)` | Initial heading between two points |
| `midpoint(lat1, lon1, lat2, lon2)` | Great-circle midpoint |

**Dependencies:** numpy, scipy.linalg, dataclasses, typing

---

### `related/shakeout_v2/fault_plots.py`
**Description:** Visualization classes for fault models and slip distributions.

| Class | Description |
|-------|-------------|
| `FaultPlot3D` | 3D fault visualization (map, outlines, vectors, slip patches) |
| `FaultPlot2D` | 2D plan-view fault visualization |

Methods include: `showmap`, `set_zlim`, `set_lims`, `plot_outlines`, `plot_shapefile`, `plot_symbols`, `plot_vectors`, `plot_up_vectors`, `plot_slip_patches`.

**Dependencies:** matplotlib, numpy

---

### `related/shakeout_v2/shakeout.py`
**Description:** Utility module for Okada-based modeling with geographic data. Handles GNSS data I/O, coordinate preparation, forward modeling, misfit computation, and map visualization.

| Function | Description |
|----------|-------------|
| `read_gnss_data(fname, verbose)` | Read GNSS .vel files, return dict of arrays |
| `fault_outline_en(depth, dip, L, W, strike, E0, N0, return_depths)` | Fault outline in ENU (km) |
| `prepare_coordinates(gnss_data, lat0, lon0, alt0, verbose)` | Transform GNSS to ENU |
| `compute_okada_displacements(e_km, n_km, depth, strike, dip, L, W, rake, slip, verbose)` | Okada forward model |
| `compute_misfit(oe, on, ou, pe, pn, pu, ere, ern, eru)` | RMSE, chi-squared, reduced chi-squared |
| `fault_corners_to_latlon(corners_EN, lat0, lon0, alt0)` | Convert fault corners ENU to lat/lon |
| `create_geographic_map(gnss_data, fault_latlon, lon0, lat0, pred_E, pred_N, pred_Z, ...)` | Cartopy map with data + predictions |
| `create_comparison_plots(oe, on, ou, pe, pn, pu, rmse_E, rmse_N, rmse_Z, ...)` | Scatter plots obs vs predicted |

**Dependencies:** numpy, pandas, matplotlib, cartopy, okada85, geod_transform

---

### `related/shakeout_v2/shakeout_mcmc.py`
**Description:** MCMC Bayesian inversion utilities for fitting single-fault parameters to GNSS data.

| Function | Description |
|----------|-------------|
| `fault_model_for_fitting(gps_locs, *params)` | Predict GPS displacements from 9 fault parameters |
| `format_gps_data(lon, lat, oe, on, ou, ere, ern, eru)` | Format data for MCMC sampler |
| `lnlike_fault(params, x, y, yerr)` | Log-likelihood for Gaussian noise model |
| `lnprior_fault(params, x, y, yerr, minvals, maxvals)` | Log-prior with uniform bounds |
| `lnprob_fault(params, x, y, yerr, minvals, maxvals)` | Log-posterior = prior + likelihood |

**Dependencies:** numpy, okada85, geod_transform

---

### `related/shakeout_v2/euler_calc.py`
**Description:** Euler pole calculations for plate motion.

| Function | Description |
|----------|-------------|
| `convert_velocity(lat, lon, v_e, v_n)` | Convert ENU velocity to ECEF |
| `best_fit_pole(lat, lon, v_e, v_n)` | Find best-fit Euler pole from station velocities |
| `pole_velocity(lat, lon, pole_lat, pole_lon, pole_rate)` | Predict velocity at a point from an Euler pole |
| `euler_vector(lat_p, lon_p, deg_myr)` | Convert pole location + rate to Cartesian angular velocity |
| `euler_location(omega)` | Convert angular velocity vector to pole location + rate |
| `euler_jacobian(lat, lon, omega)` | Jacobian of velocity w.r.t. Euler vector |
| `get_2d_covar_mat(lat, lon, omega, cov_omega)` | Propagate uncertainty to velocity |
| `euler_rot_matrix(lat, lon)` | Rotation matrix for Euler velocity at a point |

**Dependencies:** numpy, scipy.linalg, geod_transform

---

### `related/shakeout_v2/moment_tensor.py`
**Description:** Moment tensor computation from fault geometry.

| Function | Description |
|----------|-------------|
| `get_moment_tensor(strike, dip, rake, M0)` | Compute 3x3 moment tensor from strike/dip/rake |
| `get_moment(L, W, slip, mu)` | Compute scalar seismic moment |
| `get_magnitude(L, W, slip)` | Compute moment magnitude Mw |
| `print_moment_tensor(MT)` | Pretty-print moment tensor |
| `save_moment_tensor(MT, fname)` | Save to file |

**Dependencies:** numpy

---

### Other Python Files

| File | Description |
|------|-------------|
| `related/shakeout_v2/okada85.py` | Copy of `geometry/okada/okada85.py` for local use |
| `related/shakeout_v2/okada_greens.py` | Copy of `geometry/okada/okada_greens.py` |
| `related/shakeout_v2/my_shapefile.py` | Shapefile reader/writer (Reader, Writer, Editor classes) |
| `related/shakeout_v2/sf_slipmodel.py` | Empty file (no functions) |

---

### Scripts

| File | Description |
|------|-------------|
| `related/shakeout_v2/shakeout_sim.py` | Script loading fault model and SuGAR/SuMo sites for simulation |
| `related/shakeout_v2/compare_slip_models.py` | Script comparing two slip models |
| `related/shakeout_v2/mentawai_slipmodel.py` | Script generating a Mentawai earthquake slip model |

---

### Test Files

| File | Description |
|------|-------------|
| `related/shakeout_v2/test.py` | General test script |
| `related/shakeout_v2/test_faultmodel.py` | Tests for FaultModel class |
| `related/shakeout_v2/test_faultplots.py` | Tests for FaultPlot classes |
| `related/shakeout_v2/test_geod.py` | Geodetic function tests |
| `related/shakeout_v2/test_geod_transform.py` | Comprehensive geod_transform tests |
| `related/shakeout_v2/test_laplacian.py` | Laplacian smoothing operator tests |
| `related/shakeout_v2/test_okada85.py` | Okada85 regression tests |
| `related/shakeout_v2/test_type_scratch.py` | Type-checking scratch tests |

---

## 2. Tutorial Notebooks (`related/shakeout_v2/notebooks/`)

A progressive sequence (00-08) teaching linear and nonlinear geophysical inverse theory, from basic least squares through regularized multi-dataset inversion and MCMC.

### Notebook 00: Setup and Data Overview
**File:** `notebooks/00_setup_and_data_overview.ipynb`

**Purpose:** Environment setup, load synthetic GNSS site locations, introduce d=Gm notation. Establishes the observation grid and data format used in subsequent notebooks.

**Dependencies:** numpy, matplotlib

---

### Notebook 01: Least Squares from the Mean
**File:** `notebooks/01_least_squares_from_mean.ipynb`

**Purpose:** Demonstrates that minimizing squared residuals yields the sample mean. Motivates least squares as a principled estimator.

**Dependencies:** numpy, matplotlib

---

### Notebook 02: Line Fitting and the Design Matrix
**File:** `notebooks/02_line_fitting_and_design_matrix.ipynb`

**Purpose:** Build a design matrix for y = mx + b. Solve with `np.linalg.lstsq`. Interpret residuals and estimate parameter uncertainty from Cm = s^2 (G^T G)^{-1}.

**Key concepts:** Design matrix construction, least-squares normal equations, residual analysis, parameter covariance.

**Dependencies:** numpy, matplotlib

---

### Notebook 03: Okada Forward Model (Single Fault)
**File:** `notebooks/03_okada_forward_model_single_fault.ipynb`

**Purpose:** Use `okada85.displacement()` as a forward model. Plot fault outline on a map, horizontal displacement vectors, and vertical filled circles.

**Key functions used:** `okada85.displacement`, `fault_outline`, `plot_fault_patches`, `plot_data_on_map` (defined in notebook).

**Dependencies:** numpy, matplotlib, okada85, okada_utils

---

### Notebook 04: Fault Discretization and G Matrix
**File:** `notebooks/04_fault_discretization_and_G_matrix.ipynb`

**Purpose:** Subdivide a single fault into multiple patches. Build the full Green's function matrix G. Perform unregularized least-squares inversion and observe instability.

**Key functions used:** `build_patch_grid`, `build_component_greens`, `plot_fault_patches`.

**Dependencies:** numpy, matplotlib, okada85, okada_utils

---

### Notebook 05: Regularization -- Ridge and Smoothing
**File:** `notebooks/05_regularization_ridge_and_smoothing.ipynb`

**Purpose:** Apply ridge regression (Tikhonov zeroth-order), Laplacian smoothing (second-order), and non-negative bounded least squares (scipy `lsq_linear`). Compare regularized solutions to unregularized.

**Key concepts:** Augmented system [G; lambda*L], damping parameter lambda, Laplacian smoothing operator, bounded least squares.

**Dependencies:** numpy, matplotlib, scipy.optimize, okada85, okada_utils

---

### Notebook 06: Choosing the Regularization Parameter
**File:** `notebooks/06_choosing_regularization_parameter.ipynb`

**Purpose:** L-curve method (maximum curvature) and ABIC (Akaike Bayesian Information Criterion) for selecting optimal lambda. Compares results from both methods.

**Key concepts:** L-curve (log misfit vs log model norm), curvature computation, ABIC formula, optimal regularization selection.

**Dependencies:** numpy, matplotlib, scipy.optimize, okada85, okada_utils

---

### Notebook 07: Weighted Least Squares with Multiple Datasets
**File:** `notebooks/07_weighted_least_squares_multiple_datasets.ipynb`

**Purpose:** Weighted least squares with block-diagonal covariance. Combine datasets with different noise levels. Demonstrates N_k normalization for dataset balancing.

**Key concepts:** WLS formula m = (G^T Cd^{-1} G)^{-1} G^T Cd^{-1} d, block-diagonal covariance, dataset weighting.

**Dependencies:** numpy, matplotlib, scipy.linalg

---

### Notebook 07b: GNSS + InSAR Joint Inversion with Correlated Noise
**File:** `notebooks/07b_gnss_insar_correlated_noise.ipynb`

**Purpose:** Three weighting strategies for combining GNSS (15 independent stations) and InSAR (1500 correlated pixels): (a) naive per-point, (b) N_k normalization, (c) full covariance C^{-1}. Shows that spatially correlated InSAR noise means more pixels does not equal more information. Computes effective number of independent observations.

**Key concepts:** Squared-exponential covariance kernel, effective N_eff ~ L/l, precision matrix, information content comparison, varying-l analysis.

**Dependencies:** numpy, matplotlib, scipy.linalg

---

### Notebook 08: Nonlinear Inversion (Scipy and MCMC)
**File:** `notebooks/08_nonlinear_inversion_scipy_and_mcmc.ipynb`

**Purpose:** Fit nonlinear model y = a*exp(-x/tau) + c. Part A: Grid search over 3D parameter space with 2D marginal misfit visualization. Part B: Deterministic optimization (Newton-CG and L-BFGS-B) with optimizer path on misfit surface. Part C: MCMC (emcee) Bayesian sampling with corner plots of posterior distributions.

**Key concepts:** Grid search, Newton's method with analytical gradient, L-BFGS-B bounded optimization, log-likelihood/prior/posterior, affine-invariant ensemble sampler, corner plots, burn-in, credible intervals.

**Dependencies:** numpy, matplotlib, scipy.optimize, emcee, corner

---

## 3. Standalone Notebooks (`related/shakeout_v2/`)

### MCMC_notes.ipynb
**Purpose:** Course lab notebook for Bayesian analysis and Monte Carlo simulation. Two examples: (1) Line fitting with MCMC (benchmark against least squares), (2) Ridgecrest earthquake fault parameter inversion from GPS data using emcee. Also demonstrates zeus ensemble slice sampler as an alternative.

**Dependencies:** numpy, pandas, matplotlib, scipy.optimize, emcee, corner, zeus, fault_model, geod_transform, moment_tensor

---

### illustrate_fault.ipynb
**Purpose:** Generate illustrations of fault patch subdivision with Python indexing labels. Creates two visualizations: (1) a thrust fault grid with Gaussian slip and s_k labels, (2) a rectangular fault with 0-based patch indices and grayscale slip shading.

**Dependencies:** numpy, matplotlib

---

### okada_demo_shallow_thrust.ipynb
**Purpose:** Demonstrate the Okada85 displacement solution for a shallow dipping thrust fault. Visualizes fault geometry in map view, N-S profile, and E-W profile. Computes and plots displacement fields (East, North, Vertical, horizontal vectors) on a grid.

**Dependencies:** numpy, matplotlib, okada85

---

### okada_gnss_forward_model.ipynb
**Purpose:** Full workflow for Okada forward modeling with real GNSS data. Loads SuGAr Mentawai velocity data, defines a single fault patch, computes predicted displacements, creates geographic maps with Cartopy, runs 1D grid search on parameters, and performs multi-parameter Nelder-Mead optimization.

**Dependencies:** numpy, pandas, matplotlib, shakeout, scipy.optimize

---

### okada_mcmc_inversion.ipynb
**Purpose:** Complete MCMC Bayesian inversion of GNSS data for earthquake fault parameters. Loads SuGAr Mentawai data, configures emcee sampler (20 walkers, 400+400 iterations), analyzes convergence, generates corner plots, compares observed vs. predicted on geographic maps.

**Dependencies:** numpy, pandas, matplotlib, emcee, corner, shakeout, shakeout_mcmc

---

### plot_mentawai_displacement.ipynb
**Purpose:** Plot coseismic displacement vectors from SuGAr GPS data on a Cartopy geographic map. Visualizes horizontal vectors and vertical motion (colored circles) for the Mentawai earthquake.

**Dependencies:** numpy, pandas, matplotlib, cartopy, geod_transform

---

### test_faultmodel.ipynb
**Purpose:** Interactive testing of the FaultModel class. Creates a model instance and tests basic functionality.

**Dependencies:** fault_model, fault_plots, geod_transform, numpy

---

## 4. Notebook-Local Copies

| File | Same as |
|------|---------|
| `notebooks/okada85.py` | `geometry/okada/okada85.py` |
| `notebooks/okada_utils.py` | `geometry/okada/okada_utils.py` |
