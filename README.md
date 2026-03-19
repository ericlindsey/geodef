# GeoDef

A flexible, student-friendly Python library for forward and inverse modeling of fault slip in elastic half-spaces. Targets both coseismic (earthquake) and interseismic (locked fault / coupling) applications.

## Current Status

**Phase 1 complete** -- all three Green's function engines are implemented and verified with 113 tests:

| Engine | Geometry | Computes | Source |
|--------|----------|----------|--------|
| `okada85` | Rectangular | Surface displacement, tilt, strain | Okada (1985), ported from Matlab |
| `okada92` | Rectangular | Displacement and strain at arbitrary depth | Okada (1992), ported from Fortran DC3D |
| `tdcalc` | Triangular | Full-/half-space displacement and strain | Nikkhoo & Walter (2015), ported from Matlab |

Engines are cross-validated against each other (okada85 vs okada92 at the surface, triangular pairs vs rectangles, etc.).

## Planned

The library will provide a simple, layered API for geodetic modeling:

- **Fault geometry** -- create, discretize, and visualize rectangular and triangular fault meshes
- **Data containers** -- GNSS, InSAR, and other geodetic data types with a common interface
- **Green's matrix assembly** -- polymorphic over fault type and data type
- **Inversion** -- regularized least-squares with automatic hyperparameter tuning (ABIC, cross-validation)
- **Uncertainty** -- model covariance, resolution matrices, fit statistics

See `PLAN.md` for the full development roadmap.

## Installation

```bash
uv pip install -e .
```

## Testing

```bash
uv run pytest
```

## References

- Okada, Y., 1985. Surface deformation due to shear and tensile faults in a half-space. *Bull. Seismol. Soc. Am.*, 75(4), 1135--1154.
- Okada, Y., 1992. Internal deformation due to shear and tensile faults in a half-space. *Bull. Seismol. Soc. Am.*, 82(2), 1018--1040.
- Nikkhoo, M. & Walter, T.R., 2015. Triangular dislocation: an analytical, artefact-free solution. *Geophys. J. Int.*, 201(2), 1119--1141.
