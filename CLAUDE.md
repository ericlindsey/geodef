# CLAUDE.md - Agent Onboarding Guide

## Project Overview

**GeoDef** is being built as a flexible, student-friendly Python library for **forward and inverse modeling of fault slip** in elastic half-spaces. It targets both coseismic (earthquake) and interseismic (locked fault / coupling) applications.

The project name is `geodef`. The future package will be installed as `geodef`.

## Current Repository Layout

```
geodef/
├── CLAUDE.md              # This file — agent onboarding
├── PLAN.md                # Development roadmap
├── PYTHON.md              # Mandatory coding standards
├── docs/                  # Auto-generated code overviews
│   ├── overview_greens_and_geometry.md
│   ├── overview_examples_and_notebooks.md
│   └── overview_stress_shadows.md
├── tests/                 # Consolidated test suite (113 tests, run with `uv run pytest`)
│   ├── test_okada85.py    # Okada85 reference cases + property tests
│   ├── test_okada92.py    # Okada92/DC3D tests
│   ├── test_tdcalc.py     # Triangular dislocation tests
│   ├── test_cross_validation.py  # Cross-validation between all engines
│   └── reference_data/    # Matlab-generated .npz reference files
├── geometry/              # Core Green's functions & mesh tools (Python + Matlab/Fortran sources)
│   ├── okada/             # Okada85/92 rectangular dislocations
│   ├── tdcalc/            # Triangular dislocations (Nikkhoo & Walter 2015)
│   └── slabMesh/          # Triangular mesh generation from slab2.0 grids
└── related/               # Reference code and teaching materials
    ├── shakeout_v2/       # Python fault modeling classes, plotting, inversions, tutorials
    │   ├── notebooks/     # Numbered tutorial series (00–08) on geodetic inversion
    │   └── *.py           # FaultModel, SlipModel, geod_transform, euler_calc, etc.
    └── stress-shadows/    # Matlab inversion framework (Lindsey et al., 2021)
        ├── objects/       # OOP inversion: Jointinv, Sources, Datasets
        ├── functions/     # Regularization, ABIC, plotting, coordinate transforms
        ├── unicycle/      # Green's functions, geometry, ODE/earthquake-cycle modeling
        └── *_models/      # 2D/3D synthetic examples + real applications (Nepal, Japan, Cascadia)
```

See `docs/` for detailed file-level and function-level overviews of each area.

## Green's Functions (Forward Models)

| Module | Geometry | What it computes | Language | Status |
|--------|----------|-----------------|----------|--------|
| `geometry/okada/okada85.py` | Rectangular | Surface displacements, tilts, strains | Python (from Matlab) | Verified (27 ref + 17 property tests) |
| `geometry/okada/okada92.py` | Rectangular | Internal deformation at depth (displacements, strains) | Python (from Fortran dc3d.f90) | Verified (10 tests + cross-validated vs okada85) |
| `geometry/tdcalc/tdcalc.py` | Triangular | Full-/half-space displacements and strains | Python (from Matlab) | Verified (8 ref + 4 property + cross-validated) |
| `geometry/okada/okada_greens.py` | Rectangular | Green's matrix assembly (G for displacement/strain) | Python | Working |
| `geometry/okada/okada_utils.py` | Rectangular | Fault outlines, patch grids, Laplacian, component greens | Python | Working |
| `geometry/slabMesh/slabMesh.py` | Triangular | Mesh generation from slab2.0 NetCDF grids | Python | Working |

## Key Python Modules in `related/shakeout_v2/`

| Module | Purpose |
|--------|---------|
| `fault_model.py` | `FaultModel` class — patch management, Green's function dispatch, geometry |
| `slip_model.py` | `SlipModel` class — slip assignment, moment calculation, forward modeling |
| `geod_transform.py` | Geodetic transforms: ECEF, ENU, geodetic, Vincenty, haversine |
| `okada_greens.py` | Green's matrix assembly (copy of `geometry/okada/okada_greens.py`) |
| `euler_calc.py` | Euler pole fitting and velocity predictions |
| `moment_tensor.py` | Moment tensor computation from strike/dip/rake |
| `fault_plots.py` | `FaultPlot3D`/`FaultPlot2D` visualization classes |
| `shakeout.py` | Utilities for Okada modeling with geographic coordinates |
| `shakeout_mcmc.py` | MCMC Bayesian inversion for fault parameters |

## Tutorial Notebooks (`related/shakeout_v2/notebooks/`)

A progressive series building from basic statistics to full geodetic inversion:

| # | Topic | Key concepts |
|---|-------|-------------|
| 00 | Setup & data overview | Environment, synthetic GNSS data, d=Gm notation |
| 01 | Least squares from mean | Sample mean as least-squares solution |
| 02 | Line fitting | Design matrix, uncertainty propagation |
| 03 | Okada forward model | Single-fault displacement, map-view plotting |
| 04 | Fault discretization | Multi-patch G matrix, unregularized inversion |
| 05 | Regularization | Ridge, Laplacian smoothing, non-negative constraints |
| 06 | Choosing lambda | L-curve curvature, ABIC |
| 07 | Weighted least squares | Block-diagonal covariance, multiple datasets, correlated noise |
| 07b | GNSS + InSAR noise | Correlated InSAR noise handling |
| 08 | Nonlinear inversion | scipy.optimize, MCMC with emcee |

## Matlab Inversion Framework (`related/stress-shadows/`)

The primary source for the inverse-modeling architecture being ported to Python:

- **Objects**: `Jointinv` (driver), `Jointinv_Source`/`Jointinv_Dataset` (abstract bases), `Static_Halfspace_Fault_Source`, `Backslip_Translation_Source`, `Static_GPS_Dataset`, `Static_LOS_Dataset`, `Static_Coral_Dataset`
- **Functions**: Regularization (`compute_laplacian.m`), hyperparameters (`abic_alphabeta.m`, `kfold_cv_jointinv.m`), coordinate transforms (`latlon_to_xy_polyconic.m`), stress kernels (`unicycle_stress_kernel.m`), plotting, I/O
- **Unicycle**: Green's functions (Okada85/92, Nikkhoo15, Meade07), geometry classes (patch, triangle, receiver), ODE solvers for earthquake-cycle modeling, rheology

## Priority & Roadmap

See `PLAN.md` for the detailed development plan.

High-level priorities:
1. ~~Finalize and test existing Green's function implementations (okada85, okada92, tdcalc)~~ **DONE** — 113 tests passing
2. Design the `geodef` package structure for maximum student usability ← **NEXT**
3. Implement core library: forward models, fault geometry, data containers
4. Implement inverse framework: G assembly, regularization, solvers, hyperparameters
5. Port tutorial notebooks to use the new library
6. Add uncertainty quantification

## Important Rules

- **Read `PYTHON.md` before editing any code.** It contains mandatory style guidelines, tooling requirements, and coding standards.
- Use red/green TDD. Write tests first, then write code to pass the tests.
- Use `uv` for package management. Use `pytest` for testing.
- All new functions must have type hints, docstrings, and tests.
- Use NumPy vectorization — avoid Python loops over observation points or fault patches.
- Coordinate convention: the library uses a local Cartesian (x=East, y=North, z=Up) frame unless otherwise noted. Green's functions may use internal conventions (e.g. Okada uses x=strike, y=updip) — always convert at the interface boundary.

## Testing

Run tests with:
```bash
uv run pytest
```

**113 tests passing** across 4 test files (949 lines):

| File | Tests | What it covers |
|------|-------|---------------|
| `tests/test_okada85.py` (254 lines) | 44 | 9 reference cases x 3 outputs (disp/tilt/strain), geometry (4 dips x 3 slip types), symmetry (3), far-field (1), vectorization (2) |
| `tests/test_okada92.py` (125 lines) | 10 | Shape, dip variations (4), slip components (3), depth variation (2), linearity (2), input validation (1) |
| `tests/test_tdcalc.py` (142 lines) | 12 | 4 Matlab reference configs x 2 (disp + strain), zero-slip, linearity, far-field decay, FS vs HS |
| `tests/test_cross_validation.py` (428 lines) | 47 | Okada85 vs DC3D surface (12), Okada85 vs Okada92 wrapper (12), tdcalc vs Okada85 surface (9), tdcalc vs DC3D depth (13 disp + 1 strain) |

Reference data: `tests/reference_data/` contains 4 `.npz` files (FS_simple, FS_complex, HS_simple, HS_complex) extracted from Matlab tdcalc.

Additional test files (not run by default via `uv run pytest`):
- `related/shakeout_v2/test_geod_transform.py` — Coordinate transformation tests
- `related/shakeout_v2/test_laplacian.py` — Laplacian smoothing operator tests
- `related/shakeout_v2/test_faultmodel.py` — FaultModel class tests

## References

- Okada, Y., 1985. Surface deformation due to shear and tensile faults in a half-space. BSSA.
- Okada, Y., 1992. Internal deformation due to shear and tensile faults in a half-space. BSSA.
- Nikkhoo, M., & Walter, T.R., 2015. Triangular dislocation: an analytical, artefact-free solution. GJI.
- Lindsey, E.O. et al., 2021. Slip rate deficit and earthquake potential on shallow megathrusts. Nature Geoscience.
