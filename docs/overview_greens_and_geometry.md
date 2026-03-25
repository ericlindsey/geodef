# Green's Functions and Geometry Overview

Covers: `geometry/okada/`, `geometry/tdcalc/`, `geometry/slabMesh/`

---

## 1. geometry/okada/

Rectangular dislocation models (Okada 1985/1992) for computing surface and internal deformation in an elastic half-space.

### Python Modules

#### `geometry/okada/okada85.py`
**Description:** Core Python implementation of the Okada (1985) analytical solution for surface displacement, tilt, and strain due to a finite rectangular fault in a homogeneous elastic half-space.

| Function | Description |
|----------|-------------|
| `setup_args(obs_x, obs_y, depth, strike, dip, L, W, nu)` | Validate/broadcast inputs, rotate coordinates into Okada frame |
| `displacement(obs_x, obs_y, depth, strike, dip, L, W, rake, slip, opening, nu)` | Surface displacement (East, North, Up) |
| `tilt(obs_x, obs_y, depth, strike, dip, L, W, rake, slip, opening, nu)` | Surface tilt |
| `strain(obs_x, obs_y, depth, strike, dip, L, W, rake, slip, opening, nu)` | Surface strain tensor components |
| `chinnery(f, ...)` | Chinnery's notation for combining corner contributions |
| `ux_ss, uy_ss, uz_ss` | Strike-slip displacement sub-functions |
| `ux_ds, uy_ds, uz_ds` | Dip-slip displacement sub-functions |
| `ux_tf, uy_tf, uz_tf` | Tensile fault displacement sub-functions |
| `I1, I2, I3, I4, I5` | Auxiliary Chinnery integrals |
| `K1, K2, K3` | Tilt sub-functions |
| `J1, J2, J3, J4` | Strain sub-functions |

**Dependencies:** numpy

---

#### `geometry/okada/okada92.py`
**Description:** Python port of the Okada (1992) DC3D Fortran subroutine. Computes displacement and derivatives at arbitrary depth (not just surface) due to a rectangular dislocation in a half-space.

| Function | Description |
|----------|-------------|
| `okada92(obs_x, obs_y, obs_depth, ...)` | Wrapper: rotates into strike coordinates, calls DC3D, rotates back |
| `DC3D(ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3)` | Main DC3D calculation (displacement + 9 derivatives) |
| `UA(XI, ET, Q, DISL1, DISL2, DISL3)` | Displacement contribution Part A |
| `UB(XI, ET, Q, DISL1, DISL2, DISL3)` | Displacement contribution Part B |
| `UC(XI, ET, Q, Z, DISL1, DISL2, DISL3)` | Displacement contribution Part C (image) |
| `DCCON0(ALPHA, DIP)` | Set medium constants (Poisson ratio, dip trig) |
| `DCCON2(XI, ET, Q, SD, CD, KXI, KET)` | Station-geometry constants |

**Dependencies:** numpy, math. Uses global variables for intermediate state.

---

#### `geometry/okada/okada_greens.py`
**Description:** Builds Green's function matrices for displacement and strain from arrays of fault patches and observation points. Handles geographic-to-local coordinate conversion.

| Function | Description |
|----------|-------------|
| `displacement_greens(lat, lon, lat0, lon0, depth, strike, dip, L, W, nu)` | Build G matrix: rows = 3*N_obs (E,N,U per station), cols = 2*N_patches (ss,ds) |
| `strain_greens(lat, lon, lat0, lon0, depth, strike, dip, L, W, nu)` | Build G matrix for surface strain |
| `resolution(G)` | Compute resolution matrix R = G(G^TG)^{-1}G^T |
| `check_lengths(...)` | Validate input array lengths match |

**Dependencies:** sys, numpy, okada85, geod_transform

---

#### `geometry/okada/okada_utils.py`
**Description:** Utility functions for fault geometry calculations, patch grid generation, and Laplacian smoothing operators.

| Function | Description |
|----------|-------------|
| `fault_outline(depth_m, dip_deg, length_m, width_m, strike_deg, centroid_E_m, centroid_N_m, return_depths)` | Compute fault outline polygon in local E/N coordinates |
| `build_patch_grid(e0, n0, z0, strike_deg, dip_deg, fault_L, fault_W, nL, nW)` | Subdivide a fault into nL x nW rectangular patches; returns list of patch dicts |
| `build_component_greens(obs_e, obs_n, patches, rake_deg, nu)` | Build G matrix for a list of patches at fixed rake |
| `build_laplacian_2d(nL, nW)` | 2D finite-difference Laplacian smoothing operator |
| `build_laplacian_2d_simple(nL, nW)` | Simplified Laplacian (no edge weighting) |

**Dependencies:** numpy, okada85

---

### Matlab Modules

#### `geometry/okada/okada85.m`
**Description:** Matlab implementation of Okada (1985) by F. Beauducel. Full solution for displacement, tilt, and strain with optional 3D visualization. Handles vectorized fault and observation arrays.

**Key functions (internal):** `ux_ss`, `uy_ss`, `uz_ss`, `ux_ds`, `uy_ds`, `uz_ds`, `ux_tf`, `uy_tf`, `uz_tf`, chinnery integrals.

---

#### `geometry/okada/computeOkada92.m`
**Description:** Matlab implementation of Okada (1992) for internal deformation. Computes displacement, spatial derivatives, strain, and stress at depth. Includes FA/FAD/FB/FBD/FC/FCD sub-functions and COMMONPARA/DISPPARA/DERIPARA helpers.

---

#### `geometry/okada/okada85_checklist.m`
**Description:** Validation script comparing Matlab `okada85.m` output against published Table 2 checklist values from Okada (1985).

---

#### `geometry/okada/test_okada92.m`
**Description:** Simple test script for `computeOkada92`.

---

### Fortran Source

#### `geometry/okada/dc3d.f90`
**Description:** Original Fortran DC3D subroutine by Y. Okada (1991, revised 2002). Computes displacement and its spatial derivatives at depth for a finite rectangular source.

#### `geometry/okada/dc3d0.f90`
**Description:** Original Fortran DC3D0 subroutine for a point source dislocation.

---

### Test Files (Python)

The primary test suite is in `tests/` at the repository root (run with `uv run pytest`):

| File | Tests | Description |
|------|-------|-------------|
| `tests/test_okada85.py` | 44 | 9 reference cases (Table 2) x disp/tilt/strain, geometry across dips and slip types, symmetry, far-field, vectorization |
| `tests/test_okada92.py` | 10 | Dip variations, slip components, depth variation, linearity, input validation |
| `tests/test_tdcalc.py` | 12 | 4 Matlab reference configs x disp/strain, zero-slip, linearity, decay, FS vs HS |
| `tests/test_cross_validation.py` | 47 | Okada85 vs DC3D (surface), Okada85 vs Okada92 wrapper, tdcalc vs Okada85, tdcalc vs DC3D (depth) |

Reference data in `tests/reference_data/`: FS_simple.npz, FS_complex.npz, HS_simple.npz, HS_complex.npz (from Matlab tdcalc).

Legacy test files (in `geometry/okada/`, not part of the main test suite):

| File | Description |
|------|-------------|
| `geometry/okada/test_okada85.py` | Original unit tests against Okada (1985) Table 2 checklist values |
| `geometry/okada/test_okada92.py` | Original tests for the `okada92` Python wrapper |
| `geometry/okada/test_DC3D.py` | Original tests for the `DC3D` function directly |
| `geometry/okada/test_okada_wrapper.py` | Cross-validation tests using third-party `okada_wrapper` (dc3dwrapper) |

---

## 2. geometry/tdcalc/

Triangular dislocation element (TDE) code, ported from Nikkhoo & Walter (2015) Matlab implementation. Provides an artefact-free analytical solution for displacement and strain from triangular fault patches.

### Python Module

#### `geometry/tdcalc/tdcalc.py`
**Description:** Full Python implementation of the Nikkhoo & Walter (2015) TDE solution. Computes displacement and strain at surface or depth for triangular dislocation elements in a half-space or full-space.

| Function | Description |
|----------|-------------|
| `TDdispHS(X, Y, Z, P1, P2, P3, Ss, Ds, Ts, nu)` | Half-space displacement from a triangular dislocation |
| `TDdispFS(X, Y, Z, P1, P2, P3, Ss, Ds, Ts, nu)` | Full-space displacement |
| `TDstrainHS(X, Y, Z, P1, P2, P3, Ss, Ds, Ts, nu)` | Half-space strain tensor |
| `TDstrainFS(X, Y, Z, P1, P2, P3, Ss, Ds, Ts, nu)` | Full-space strain tensor |
| `strain2stress(Strain, mu, lambda_)` | Convert strain to stress tensor (Hooke's law) |
| `TDSetupD(...)` | Setup for displacement calculation at one vertex |
| `TDSetupS(...)` | Setup for strain calculation at one vertex |
| `AngDisDisp(...)` | Angular dislocation displacement |
| `AngDisStrain(...)` | Angular dislocation strain |
| `AngSetupFSC(...)` | Full-space complement angular setup |
| `AngSetupFSC_S(...)` | Full-space complement for strain |
| `AngDisDispFSC(...)` | Angular dislocation displacement (full-space complement) |
| `AngDisStrainFSC(...)` | Angular dislocation strain (full-space complement) |
| `trimodefinder(...)` | Determine observation point mode (inside/outside/on edge) |
| `setupTDCS(...)` | Set up coordinate system for a triangle |
| `TensTrans(...)` | Tensor coordinate transformation |
| `CoordTrans(...)` | Vector coordinate transformation |
| `normalize(v)` | Normalize a vector |
| `build_tri_coordinate_system(P1, P2, P3)` | Build local coordinate system for triangle |

**Dependencies:** numpy

---

### Test and Demo Files

Primary tdcalc tests are in `tests/test_tdcalc.py` (12 tests against .npz reference data) and `tests/test_cross_validation.py` (cross-validation vs Okada85/DC3D).

Legacy files:

| File | Description |
|------|-------------|
| `geometry/tdcalc/test_tdcalc.py` | Original tests against Matlab reference values stored in .mat files |
| `geometry/tdcalc/run_tdcalc.ipynb` | Demo notebook running tests, showing example displacement/strain calculations |

### Matlab Source

| File | Description |
|------|-------------|
| `geometry/tdcalc/matlab_source/TDdispFS.m` | Full-space displacement (original Matlab) |
| `geometry/tdcalc/matlab_source/TDdispHS.m` | Half-space displacement (original Matlab) |
| `geometry/tdcalc/matlab_source/TDstressFS.m` | Full-space stress (original Matlab) |
| `geometry/tdcalc/matlab_source/TDstressHS.m` | Half-space stress (original Matlab) |
| `geometry/tdcalc/matlab_source/gen_test_data.m` | Generate reference test data (.mat files) |

---

## 3. geometry/slabMesh/

Triangular mesh generation for subduction zone fault surfaces, built from Slab2.0 NetCDF grids.

### Python Module

#### `geometry/slabMesh/slabMesh.py`
**Description:** Functions for loading Slab2.0 grids, cropping regions, generating triangular meshes with variable refinement, and exporting mesh formats.

| Function | Description |
|----------|-------------|
| `load_slab2_grid(fname)` | Load a Slab2.0 NetCDF file, return X, Y, Z grids |
| `crop_rectangle(X, Y, Z, lonmin, lonmax, latmin, latmax)` | Crop grid to rectangular region |
| `slabtop_at_zero(Z)` | Shift depth values so slab top is at z=0 |
| `my_griddata(xin, yin, zin, xout, yout)` | Interpolation with NaN handling |
| `round_trip_connect(start, end)` | Generate closed polygon edges from point indices |
| `get_simple_refinement(threshold, max_refinements)` | Uniform refinement function for meshpy |
| `get_depth_based_refinement(depth_interp, base_threshold, factor, max_refinements)` | Depth-variable refinement (finer mesh at shallow depth) |
| `fix_vertical_edges(X, Y, tolerance, offset)` | Fix near-vertical edges for mesh generation |
| `make_mesh(Xpoly, Ypoly, threshold, max_refinements)` | Generate triangular mesh with uniform refinement |
| `make_depth_variable_mesh(Xpoly, Ypoly, depth_interp, threshold, factor, max_refinements)` | Generate mesh with depth-dependent refinement |
| `save_mesh_for_blocks(pts, tris, fname)` | Export mesh in Blocks format |
| `save_mesh_for_unicycle(pts, tris, fname)` | Export mesh in Unicycle format |

**Dependencies:** numpy, scipy, netCDF4, meshpy.triangle

---

### Notebooks

| File | Description |
|------|-------------|
| `geometry/slabMesh/make_mesh.ipynb` | Creates triangular mesh for Myanmar/Rakhine megathrust from Slab2.0 data |
| `geometry/slabMesh/make_sagaing_mesh.ipynb` | Creates vertical triangular mesh for the Sagaing fault using depth-variable refinement |
| `geometry/slabMesh/test_griddata.ipynb` | Tests NaN-handling griddata interpolation helper |

### Matlab

| File | Description |
|------|-------------|
| `geometry/slabMesh/plot_mesh.m` | Matlab mesh plotting utility |
