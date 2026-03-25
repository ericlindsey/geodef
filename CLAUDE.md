# CLAUDE.md - Agent Onboarding Guide

## Project Overview

**GeoDef** is being built as a flexible, student-friendly Python library for **forward and inverse modeling of fault slip** in elastic half-spaces. It targets both coseismic (earthquake) and interseismic (locked fault / coupling) applications.

The project name is `geodef`. Install with `uv pip install -e .`.

## Current Repository Layout

```
geodef/
├── CLAUDE.md              # This file — agent onboarding
├── PLAN.md                # Development roadmap
├── PYTHON.md              # Mandatory coding standards
├── pyproject.toml         # Package config (hatchling, src layout)
├── src/geodef/            # The installable package
│   ├── __init__.py        # Top-level convenience API + version
│   ├── okada.py           # Unified dispatcher: auto-selects okada85 or okada92
│   ├── okada85.py         # Okada (1985) — surface displacements, tilts, strains
│   ├── okada92.py         # Okada (1992) — internal deformation at depth
│   ├── tri.py             # Triangular dislocation interface (Nikkhoo & Walter 2015)
│   ├── greens.py          # Green's matrix assembly + Laplacian regularization
│   ├── fault.py           # Fault class: factory methods, forward modeling, vertices, I/O
│   ├── data.py            # DataSet base + GNSS, InSAR, Vertical data types
│   ├── transforms.py      # Coordinate transforms (geographic <-> local Cartesian)
│   └── mesh.py            # Triangular mesh generation from slab2.0 (optional deps)
├── tests/                 # Test suite (309 tests, run with `uv run pytest`)
│   ├── test_okada85.py    # Okada85 reference cases + property tests
│   ├── test_okada92.py    # Okada92/DC3D tests
│   ├── test_tdcalc.py     # Triangular dislocation tests
│   ├── test_cross_validation.py  # Cross-validation between all engines
│   ├── test_package.py    # Package structure, imports, okada dispatcher
│   ├── test_transforms.py # Coordinate transformation tests
│   ├── test_greens.py     # Laplacian operator tests
│   ├── test_fault.py      # Fault construction, forward modeling, I/O, vertices
│   ├── test_data.py       # DataSet base, GNSS, InSAR, Vertical, covariance
│   ├── test_greens_integration.py  # Green's matrix assembly, projection, stacking
│   └── reference_data/    # Matlab-generated .npz reference files
├── examples/              # Worked example notebooks
│   └── 01_forward_model.ipynb  # Forward modeling demo: fault, GNSS, G matrix, prediction
├── docs/                  # Auto-generated code overviews
├── geometry/              # Original Green's function sources (Matlab/Fortran/Python)
│   ├── okada/             # Okada85/92 rectangular dislocations (originals)
│   ├── tdcalc/            # Triangular dislocations (originals)
│   └── slabMesh/          # Mesh generation (original)
└── related/               # Reference code and teaching materials
    ├── shakeout_v2/       # Python fault modeling classes, plotting, inversions, tutorials
    │   ├── notebooks/     # Numbered tutorial series (00-08) on geodetic inversion
    │   └── *.py           # Original FaultModel, SlipModel, geod_transform, etc.
    └── stress-shadows/    # Matlab inversion framework (Lindsey et al., 2021)
```

See `docs/` for detailed file-level and function-level overviews of the reference code.

## Package Modules (`src/geodef/`)

| Module | What it provides | Status |
|--------|-----------------|--------|
| `okada85` | Surface displacements, tilts, strains (Okada 1985) | Verified (44 tests) |
| `okada92` | Internal deformation at depth (Okada 1992 / DC3D) | Verified (10 tests + cross-validated) |
| `tri` | Triangular dislocation FS/HS displacements and strains | Verified (12 tests + cross-validated) |
| `okada` | Unified dispatcher: auto-selects okada85 (z=0) or okada92 (z<0) | Verified (7 tests) |
| `greens` | Green's matrix assembly, projection, stacking, Laplacian operators | Redesigned (13 + 32 tests) |
| `fault` | `Fault` class: planar/file/seg creation, forward modeling, vertices, moment, I/O | Redesigned (59 tests) |
| `data` | `DataSet` base + `GNSS`, `InSAR`, `Vertical` data types | New (47 tests) |
| `transforms` | Geodetic transforms: ECEF, ENU, geodetic, Vincenty, haversine | Migrated (19 tests) |
| `mesh` | Triangular mesh generation from slab2.0 NetCDF grids | Migrated (requires optional deps) |

## Modules Not Yet Migrated (`related/shakeout_v2/`)

| Module | Purpose |
|--------|---------|
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
1. ~~Finalize and test existing Green's function implementations (okada85, okada92, tdcalc)~~ **DONE**
2. ~~Design the `geodef` package structure and migrate existing code~~ **DONE**
3. ~~Implement core library: Fault/Data/Greens abstractions (Phase 3)~~ **DONE** (3.1–3.3, 309 tests passing)
   - ~~3.1 `Fault` class~~ **DONE** — factory classmethods, forward modeling, vertices, moment, seg format I/O
   - ~~3.2 `DataSet` classes~~ **DONE** — GNSS (3-comp or horizontal-only), InSAR (LOS), Vertical; file I/O, covariance
   - ~~3.3 `greens` assembly~~ **DONE** — polymorphic G matrix, okada+tri engines, projection, stacking helpers
4. Implement inverse framework: regularization, solvers, hyperparameters
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

**309 tests passing** across 10 test files:

| File | Tests | What it covers |
|------|-------|---------------|
| `tests/test_okada85.py` | 45 | 9 reference cases x 3 outputs (disp/tilt/strain), geometry, symmetry, far-field, vectorization |
| `tests/test_okada92.py` | 13 | Shape, dip variations, slip components, depth variation, linearity, input validation |
| `tests/test_tdcalc.py` | 12 | 4 Matlab reference configs x 2 (disp + strain), zero-slip, linearity, far-field, FS vs HS |
| `tests/test_cross_validation.py` | 43 | Okada85 vs DC3D, Okada85 vs Okada92 wrapper, tdcalc vs Okada85, tdcalc vs DC3D at depth |
| `tests/test_package.py` | 25 | Package imports, module accessibility, okada dispatcher, data class imports, API smoke tests |
| `tests/test_transforms.py` | 20 | Round-trip conversions, reference values, edge cases, vectorization, custom ellipsoids |
| `tests/test_greens.py` | 13 | Laplacian matrix shape, nullspace, stencils (interior/corner/edge), simple Laplacian |
| `tests/test_fault.py` | 59 | Fault construction, planar factory, properties, forward modeling, moment/magnitude, laplacian, file I/O (center + seg), vertices, stress kernel, cross-validation |
| `tests/test_data.py` | 47 | DataSet base, GNSS (3-comp + horizontal), InSAR (LOS projection), Vertical, covariance, file I/O |
| `tests/test_greens_integration.py` | 32 | Green's matrix assembly, single/joint datasets, okada+tri engines, projection, stacking, resolution |

Reference data: `tests/reference_data/` contains 4 `.npz` files extracted from Matlab tdcalc.

## References

- Okada, Y., 1985. Surface deformation due to shear and tensile faults in a half-space. BSSA.
- Okada, Y., 1992. Internal deformation due to shear and tensile faults in a half-space. BSSA.
- Nikkhoo, M., & Walter, T.R., 2015. Triangular dislocation: an analytical, artefact-free solution. GJI.
- Lindsey, E.O. et al., 2021. Slip rate deficit and earthquake potential on shallow megathrusts. Nature Geoscience.
