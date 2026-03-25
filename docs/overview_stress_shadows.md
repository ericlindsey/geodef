# Stress-Shadows Overview

Covers: `related/stress-shadows/` (objects, functions, examples, regional applications, unicycle)

This is a Matlab framework for joint inversion of geodetic data (GPS, InSAR, coral uplift) for fault slip, with applications to both coseismic and interseismic deformation. Published in Lindsey et al. (2021), Nature Geoscience.

---

## 1. Object Classes (`objects/`)

The inversion framework uses an object-oriented architecture with abstract base classes for sources and datasets, and a main driver class that assembles and solves the inverse problem.

### `Jointinv.m` (inherits `handle`)
**Description:** Main inversion driver. Assembles the design matrix (G), data vector (d), covariance matrix (Cd), smoothing/regularization matrices (L), constraint matrices (K), and bounds. Runs the inversion using one of three solvers.

| Property | Description |
|----------|-------------|
| `expNumber` | Experiment identifier |
| `datasets` | Cell array of `Jointinv_Dataset` objects |
| `sources` | Cell array of `Jointinv_Source` objects |
| `userParams` | Struct with experiment configuration |
| `dataVector`, `modelVector`, `predVector` | d, m, Gm vectors |
| `designMatrix` | Green's function matrix G |
| `dataCovarianceMatrix` | Block-diagonal Cd |
| `smoothingMatrix`, `smoothingVector` | Regularization L, Ld |
| `constraintMatrix`, `constraintVector` | Inequality constraints K, Kd |
| `lowerBounds`, `upperBounds` | Parameter bounds |
| `chi2` | Misfit statistic |

| Method | Description |
|--------|-------------|
| `run_setup(expfile)` | Full setup: read params, load datasets/sources, compute G, L, K |
| `read_user_params(expfile)` | Read experiment configuration from `exp_N.m` file |
| `add_datasets()` | Instantiate and load all datasets |
| `add_sources()` | Instantiate and load all source models |
| `calc_data_covariance_matrix()` | Assemble block-diagonal Cd from individual dataset covariances |
| `calc_design_matrix()` | Compute G by looping over source-dataset pairs |
| `calc_smoothing_matrix()` | Compute regularization matrix (Laplacian, stress kernel, etc.) |
| `calc_constraint_matrix()` | Compute inequality constraints |
| `calc_bounds()` | Compute parameter bounds |
| `run_inversion()` | Solve: supports `backslash`, `lsqnonneg`, `lsqlin` methods |

---

### `Jointinv_Source.m` (abstract, inherits `handle`)
**Description:** Abstract base class for all deformation sources. Defines the interface that concrete source classes must implement.

| Property | Description |
|----------|-------------|
| `fileName` | Source file name |
| `modelVector` | M x 1 model parameter vector |

Required methods for subclasses: `calc_design_matrix`, `calc_smoothing_matrix`, `calc_constraint_matrix`.

---

### `Jointinv_Dataset.m` (abstract, inherits `handle`)
**Description:** Abstract base class for all datasets. Defines the common data structure.

| Property | Description |
|----------|-------------|
| `fileName` | Data file name |
| `coordinates` | n x [2-4] observation locations |
| `dataVector` | N x 1 data vector (unraveled by point: E1,N1,U1,E2,...) |
| `predVector` | Predicted data from inversion |
| `covarianceMatrix` | N x N data covariance |
| `numComponents` | Number of data components per point |

---

### `Static_Halfspace_Fault_Source.m` (inherits `Jointinv_Source`)
**Description:** Main fault source class. Interfaces with Unicycle for Green's function and stress kernel computation. Supports rectangular (Okada92) and triangular (Nikkhoo15) fault patches.

| Property | Description |
|----------|-------------|
| `earthModel` | Unicycle greens object (okada92 or nikkhoo15) |
| `geom` | Unicycle geometry object (source or triangleReceiver) |
| `KK` | Stress kernel matrix [Kss,Kds; Ksd,Kdd] |
| `Ksn`, `Kdn` | Normal stress components |
| `rakeFile`, `Rmat`, `Vpl` | Rake rotation and plate velocity data |

| Method | Description |
|--------|-------------|
| Constructor | Loads patch file, creates Unicycle geometry and earth model |
| `calc_design_matrix(dataset, userParams, Idataset)` | G matrix for GPS (3-comp), Coral (vertical), or LOS (projected). Handles rake rotation, data component selection, InSAR trend parameters. |
| `calc_smoothing_matrix(smoothingType, userParams)` | Supports: `laplacian` (3D nearest-N), `laplacian_1d`, `stressKernel`, `sum` (total slip constraint), `value` (reference slip penalty) |
| `calc_constraint_matrix(constraintType, userParams)` | Supports: `positiveStress` (depth-dependent), `rakeSlipRate`, `rakePerpendicularSlipRate`, `negativeRakePerpendicularSlipRate` |
| `calc_bounds(bounds, userParams)` | Strike/dip/rake/rakeSlipRate bounds |

---

### `Backslip_Translation_Source.m` (inherits `Jointinv_Source`)
**Description:** Rigid translation source for points inside a polygon. Adds 2 parameters (E, N translation) to the model vector. Used for interseismic backslip models to account for block motion.

| Method | Description |
|--------|-------------|
| Constructor | Loads polygon file defining the block region |
| `calc_design_matrix(dataset, userParams)` | Creates G columns based on whether sites are inside the polygon |
| `calc_smoothing_matrix` / `calc_constraint_matrix` | No-op (not applicable) |

---

### `Static_GPS_Dataset.m` (inherits `Jointinv_Dataset`)
**Description:** 3-component GNSS displacement/velocity dataset. Reads `.vel`, `.dat`, or `.txt` file formats. Converts geographic to Cartesian coordinates via polyconic projection. Handles weights and minimum error floors.

| Property | Description |
|----------|-------------|
| `lat0`, `lon0` | Reference point for coordinate conversion |
| `name`, `lat`, `lon` | Station identifiers and locations |

---

### `Static_LOS_Dataset.m` (inherits `Jointinv_Dataset`)
**Description:** Line-of-sight (InSAR) dataset with 1 component per point. Reads `.dat` or `.txt` formats. Stores LOS look vector for projecting 3D displacements to radar line-of-sight.

| Property | Description |
|----------|-------------|
| `losVector` | n x 3 look vector (East, North, Up components) |
| `lat0`, `lon0` | LOS reference point (may differ from GPS reference) |

---

### `Static_Coral_Dataset.m` (inherits `Jointinv_Dataset`)
**Description:** Vertical-only dataset (1 component) for coral uplift rates or tide gauge data. Reads `.dat` format with weights.

---

## 2. Functions (`functions/`)

### Inversion and Hyperparameter Tuning

| Function | Description |
|----------|-------------|
| `abic_alphabeta(scenario)` | Compute ABIC for current smoothing weights (2-parameter) |
| `abic_alphabeta_sum(scenario)` | ABIC variant with sum constraint |
| `abic_smoothingonly(scenario)` | ABIC for single smoothing parameter |
| `find_best_alpha_abic(scenario)` | Optimize single regularization parameter via ABIC |
| `find_best_alphabeta_abic(scenario)` | Optimize two regularization parameters via ABIC |
| `find_best_alpha_cv(scenario, Ncv, Nk)` | Optimize regularization via cross-validation |
| `run_jointinv_abic(logweights, scenario)` | Objective function for ABIC optimization |
| `run_jointinv_cv(logweights, scenario_test, scenario_train)` | Objective function for CV optimization |
| `kfold_cv_jointinv(scenario, Nk, inv_options)` | K-fold cross-validation driver |
| `create_kfold_ind(N, k)` | Generate k-fold index partitions |
| `split_jointinv_traintest(scenario, Itest)` | Split scenario into train/test sets |
| `gridsearch_abic(scenario)` | Grid search over ABIC parameter space |
| `f_test_menke(chi2A, nA, chi2B, nB)` | F-test for model comparison |
| `model_uncertainty_chi2(scenario)` | Chi-squared uncertainty analysis |

### Regularization

| Function | Description |
|----------|-------------|
| `compute_laplacian(x, y, z, N)` | 3D finite-difference Laplacian using N nearest neighbors |
| `compute_laplacian_1d(N)` | 1D finite-difference Laplacian for line faults |

### Green's Functions and Stress Kernels

| Function | Description |
|----------|-------------|
| `unicycle_displacement_kernel(geom, coords, slipComponents, kernelFolder)` | Build 3D displacement Green's function matrix via Unicycle |
| `unicycle_stress_kernel(source, slipComponents, stressKernelFolder)` | Build stress interaction kernel via Unicycle |
| `predict_enu_model(source, ss, ds, coords, kernelFolder)` | Predict ENU displacements from slip model |

### Coordinate Transforms

| Function | Description |
|----------|-------------|
| `latlon_to_xy_polyconic(lat, lon, lat0, lon0)` | Geographic to Cartesian (polyconic projection, returns km) |
| `xy_to_latlon_polyconic(x, y, lon0, lat0)` | Cartesian to geographic (inverse polyconic) |
| `polyconic(lat, diffLon, lat0)` | Core polyconic projection function |
| `euler_vector(latp, lonp, degmyr)` | Euler pole to angular velocity vector |
| `rot_matrix(lat, lon)` | Rotation matrix for ENU at a point |

### Geometry and Mesh Utilities

| Function | Description |
|----------|-------------|
| `build_tri_from_rect(rcv, params)` | Convert rectangular patches to triangular |
| `check_triangleReceiver_signs(triangleReceiver)` | Validate triangle normal directions |
| `fix_triangleReceiver_signs(triangleReceiver)` | Fix incorrect triangle normals |
| `get_rake_rotation_matrix(rakedata)` | Rotation matrix from rake angles |
| `create_rake_file(geom, azimuth, rate, fname)` | Generate rake file from geometry |
| `write_2d_ramp(patchfname, dip, fault_width, npatch)` | Create 2D fault patch file |
| `write_3d_ramp(patchfname, xc, yc, strike, dip, ...)` | Create 3D fault patch file |
| `find_bottom(rcv)` | Find bottom edge of fault geometry |
| `get_free_patches_4x(matrix, indices)` | Extract submatrix for free patches |
| `keep_matrix_rows(A, components)` | Keep only specified component rows (E/N/U selection) |

### Data and Result Handling

| Function | Description |
|----------|-------------|
| `set_jointinv_defaults(userParams)` | Set default inversion parameters |
| `set_jointinv_path(...)` | Set paths for the framework |
| `set_unicycle_path(unicyclepath)` | Set Unicycle library path |
| `delete_jointinv_data(scenario, Idel)` | Remove data points from a scenario |
| `deuplicate_GSRM_gps_data(ingps)` | Remove duplicate GPS stations |
| `calc_coupling_result_components(scenario)` | Extract coupling results from inversion |
| `calc_moment_deficit_triangles(patchAreas, slipMag, shearModulus)` | Moment deficit from triangular patches |
| `get_moment_and_magnitude(geom, slipmag)` | Seismic moment and Mw from geometry + slip |
| `get_min_distance(xi, yi, X, Y)` | Minimum distance from point to set |

### Visualization

| Function | Description |
|----------|-------------|
| `plot_jointinv_dataset(scenario, ax, vecScale, ptScale)` | Plot all datasets (vectors + points) |
| `plot_jointinv_dataset_residual(scenario, ax, vecScale, ptScale)` | Plot data residuals |
| `plot_jointinv_dataset_3panel_residual(dataset, ax, ...)` | 3-panel residual plot |
| `plot_jointinv_slipmodel(scenario, ax, quiverScale)` | Plot slip model on fault patches |
| `plot_jointinv_slipvectors(scenario, source, ax, quiverScale, vecColor)` | Plot slip vectors |
| `plot_gridded_slip_model(scenario)` | Gridded slip model visualization |
| `plot_coupling_inversion(scenario, results, figoffset)` | Coupling inversion results |
| `plot_coupling_vectors(scenario, vecScale)` | Coupling velocity vectors |
| `scaled_quiver(x, y, v, u, scale, plotargs)` | Scaled quiver plot |
| `quiver2(...)` | Enhanced quiver plot with colorbar |
| `bluewhitered(m)` | Blue-white-red diverging colormap |
| `polarmap(...)` | Polar diverging colormap |
| `stretch_fig_no_whitespace(fig, scale)` | Remove figure whitespace |

### Export / Save

| Function | Description |
|----------|-------------|
| `save_geom_for_GMT(output_filename, geom, values, lat0, lon0)` | Export geometry for GMT plotting |
| `save_geom_for_GMT_vertfault(output_filename, geom, values, lat0, lon0)` | GMT export for vertical faults |
| `save_geom_for_interp_GMT(output_filename, geom, values, lat0, lon0)` | GMT export for interpolation |
| `save_coupling_inversion(scenario, results, descriptor)` | Save coupling results |
| `save_jointinv_model_trench0(scenario, values, fname)` | Save model with trench reference |
| `DataHash(Data, Opt)` | Compute hash of arbitrary data (third-party utility) |

### Bundled Library: mesh2d-master

A third-party Matlab triangular mesh generation library (by D. Engwirda). Provides Delaunay refinement with quality control. Key entry points: `refine2`, `smooth2`, `tridemo`, `tricost`. Contains sub-packages for AABB-tree spatial indexing, mesh utilities, geometry utilities, and mesh quality metrics.

---

## 3. Example and Application Scripts

### 2D Synthetic Models (`2d_models/`)

| File | Description |
|------|-------------|
| `exp_0.m` -- `exp_3.m` | Experiment parameter files for 2D synthetic tests |
| `forward_model_2d.m` | Generate synthetic 2D data from known slip model |
| `abic_optimize_jointinv.m` | Run ABIC optimization on 2D models |
| `fig2_model_range.m` | Generate Figure 2 (model range comparison) |
| `gridsearch_uncertainty.m` | Grid search for uncertainty quantification |
| `suppl_fig_s1_CV.m` | Supplementary figure: cross-validation |
| `suppl_fig_s2_beta_search.m` | Supplementary figure: beta parameter search |
| `suppl_fig_s2_grid_search.m` | Supplementary figure: grid search |
| `test_sum_value.m` | Test sum/value smoothing types |

### 3D Synthetic Models (`3d_models/`)

| File | Description |
|------|-------------|
| `exp_0.m`, `exp_1.m`, `exp_40.m` | 3D experiment parameter files |
| `forward_model_3d.m` | Generate synthetic 3D data |
| `run_3d_comparison.m` | Run inversion comparison (with stress penalty) |
| `run_3d_comparison_nobeta.m` | Run comparison without beta (stress smoothing) |
| `run_cv_3d.m` | Run cross-validation on 3D model |

### Example: 2D Earthquake (`example_2d_earthquake/`)

| File | Description |
|------|-------------|
| `exp_101.m` | Experiment parameters |
| `setup_example.m` | Create synthetic data for 2D earthquake |
| `run_example.m` | Run the inversion example |

### Example: 3D Earthquake (`example_3d_earthquake/`)

| File | Description |
|------|-------------|
| `exp_102.m` | Experiment parameters |
| `setup_example.m` | Create synthetic data for 3D earthquake |
| `run_inversion.m` | Run the inversion |

### Example: Nepal Earthquake (`example_nepal_earthquake/`)

| File | Description |
|------|-------------|
| `exp_401.m` -- `exp_403.m` | Experiment configurations (different regularization) |
| `setup_data_nepal.m` | Prepare Nepal GPS data |
| `run_nepal_inversion.m` | Run Nepal earthquake inversion |

### Cascadia (`cascadia/`)

| File | Description |
|------|-------------|
| `exp_0.m` -- `exp_23.m` | ~14 experiment configurations |
| `setup_data_cascadia.m` | Prepare Cascadia GPS/coral data |
| `run_cascadia_inversion.m` | Run coupling inversion |
| `run_cv_cascadia.m` | Cross-validation for Cascadia |
| `gridsearch_cascadia.m` | Grid search over parameters |

### Japan (`japan/`)

| File | Description |
|------|-------------|
| `exp_0.m` -- `exp_15.m` | ~14 experiment configurations |
| `setup_data_japan.m` | Prepare Japan GPS data |
| `run_japan_inversion.m` | Run coupling inversion |
| `run_cv_japan.m` | Cross-validation for Japan |
| `gridsearch_japan.m` | Grid search over parameters |
| `check_signs.m` | Verify triangle normal sign conventions |

---

## 4. Unicycle Library (`unicycle/matlab/+unicycle/`)

Unicycle is a Matlab library for earthquake cycle modeling, organized as a Matlab package with namespaces. It provides Green's functions, fault geometry management, ODE solvers for earthquake cycle simulations, and rheology models.

### +geometry (18 files)
Fault geometry classes for rectangular and triangular patches.

| Class/Function | Description |
|----------------|-------------|
| `source` | Rectangular fault source (reads patch files, stores geometry) |
| `receiver` | Rectangular fault receiver (for stress interactions) |
| `patch` | Base class for rectangular fault patches |
| `segment` | Fault segment definition |
| `triangle` | Base class for triangular elements |
| `triangleSource` | Triangular dislocation source |
| `triangleReceiver` | Triangular dislocation receiver |
| `trianglePassiveReceiver` | Passive triangular receiver |
| `coseismicPatch` | Coseismic rectangular patch |
| `coseismicTriangle` | Coseismic triangular patch |
| `observationPoint` | Surface observation point |
| `passiveReceiver` | Passive rectangular receiver |
| `shearZone` / `shearZoneReceiver` | Viscoelastic shear zone elements |
| `level` | Depth level definition |
| `flt2flt` | Fault-to-fault coordinate conversion |
| `shz2shz` | Shear zone-to-shear zone conversion |
| `transform4patch_general` | General patch coordinate transformation |

### +greens (34 files)
Green's function computations for multiple dislocation formulations.

| Group | Functions |
|-------|-----------|
| **Earth models** | `earthModel` (base), `okada92`, `nikkhoo15`, `gimbutas12`, `shearZone16`, `edcmp`, `model`, `stress` |
| **Displacement kernels** | `computeDisplacementOkada85`, `computeDisplacementOkada92`, `computeDisplacementNikkhoo15`, `computeDisplacementMeade07`, `computeDisplacementKernelsOkada85`, `computeDisplacementKernelsNikkhoo15`, `computeDisplacementKernelsMeade07` |
| **Stress kernels** | `computeStressKernelsOkada92`, `computeStressKernelsNikkhoo15`, `computeStressKernelsMeade07`, `computeStressKernelsGimbutas12`, `computeStressKernelsVerticalShearZone` |
| **Strain/stress** | `computeStrainMeade07`, `computeStressOkada92`, `computeStressNikkhoo15`, `computeStressPlaneStrainShearZone`, `computeStressVerticalShearZone` |
| **Traction kernels** | `computeTractionKernelsOkada92`, `computeTractionKernelsNikkhoo15`, `computeTractionKernelsVerticalShearZone` |
| **Shear zone** | `computeDisplacementAntiplaneShearZone`, `computeDisplacementVerticalShearZone`, `computeDisplacementPlaneStrainShearZone`, and TanhSinh-quadrature variants |
| **Utilities** | `computeOkada92` (core Okada92 implementation), `fBi`, `make`, `testComputeOkada85` |

### +ode (15 files)
ODE solvers and friction law implementations for earthquake cycle simulations.

| Class/Function | Description |
|----------------|-------------|
| `@evolution` | ODE evolution class with custom `ode45`, `ntrp45`, `odearguments`, `odeevents`, `odefinalize`, `odenonnegative` |
| `rateandstate` | Rate-and-state friction ODE |
| `rateandstatedamping` | Rate-and-state with radiation damping |
| `ratestrengthening` | Rate-strengthening friction |
| `ratestrengthening_prestress` | Rate-strengthening with pre-stress |
| `rateStrengtheningMaxwell` | Rate-strengthening + Maxwell rheology |
| `rateStrengtheningBurgers` | Rate-strengthening + Burgers rheology |
| `rateStrengtheningPower` | Rate-strengthening + power-law |
| `rateStrengtheningCreepTransient` | Rate-strengthening + transient creep |
| `maxwell` | Maxwell viscoelastic ODE |
| `burgers` | Burgers viscoelastic ODE |
| `eventCatalogue` | Earthquake event catalogue |
| `evt` | Single earthquake event |
| `odeset` / `odeget` | ODE solver options |
| `+hmmvp/rateandstate` | H-matrix accelerated rate-and-state |
| `+hmmvp/ratestrengthening` | H-matrix accelerated rate-strengthening |

### +manifold (3 files)
GPS observation manifold for computing synthetic observables.

| Class | Description |
|-------|-------------|
| `gps` | GPS data container and forward model |
| `gpsReceiver` | GPS receiver model |
| `+edcmp/gps` | GPS with EDCMP layered earth model |

### +optim (4 files)
Optimization and inversion utilities.

| Function | Description |
|----------|-------------|
| `compute_laplacian` | Laplacian smoothing operator for fault meshes |
| `copula` | Copula-based sampling |
| `mh` | Metropolis-Hastings MCMC sampler |
| `sim_anl` | Simulated annealing optimizer |

### +plot (2 files)
Visualization utilities.

| Function | Description |
|----------|-------------|
| `plot_faults` | Plot fault patches |
| `ellipse` | Plot error ellipses |

### +rheology (3 files)
Rheological flow laws for viscoelastic modeling.

| Class | Description |
|-------|-------------|
| `flowlaw` | Base flow law class |
| `diffusion` | Diffusion creep rheology |
| `dislocation` | Dislocation creep rheology |

### +utils (7 files)
General-purpose utilities.

| Function | Description |
|----------|-------------|
| `bisection` | Bisection root-finding |
| `computeStrainDrop` | Compute strain drop from slip |
| `computeStressDrop` | Compute stress drop from slip |
| `flt2flt` | Fault-to-fault mapping |
| `lambertW` | Lambert W function |
| `newtonRaphson` | Newton-Raphson root-finding |
| `textprogressbar` | Console progress bar |

### +export (7 files)
Data export utilities.

| Function | Description |
|----------|-------------|
| `exportVTKshearZone` | Export shear zone to VTK format |
| `exportXYZshearZone` | Export shear zone to XYZ format |
| `exportflt_rfaults` | Export faults to FLT format |
| `exportvtk_rfaults` | Export faults to VTK format |
| `exportxyz_rfaults` | Export faults to XYZ format |
| `grdread` / `grdwrite` | Read/write GMT grid files |
