# `geodef.validation` — Input validation and geometry checks

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

Two layers keep invalid physics from becoming mysterious downstream results.

## Fail-early constructor checks

Public constructors (`GNSS`, `InSAR`, `Vertical`, `Fault.planar`, ...) raise
`ValueError` at creation time for inputs that can never be right, with
messages naming the argument, its received shape or range, and the expected
unit:

- non-finite values (NaN/inf) in any numeric array or scalar;
- wrong dimensionality or mismatched lengths;
- latitudes outside [-90, 90], dips outside [0, 90], negative depths,
  non-positive lengths/widths/uncertainties;
- non-unit InSAR look vectors (pass `normalize_look=True` to renormalize);
- asymmetric or non-positive-definite covariance matrices (pass
  `validate_covariance=False` only for advanced semidefinite models).

The helpers behind these checks are public, so custom code can produce the
same style of errors:

```python
from geodef.validation import (
    as_1d_floats,       # finite 1-D float array of a required length
    check_finite_scalar,
    check_range,        # every element in [lo, hi]
    check_positive,     # strictly positive and finite
    check_covariance,   # shape, symmetry, positive definiteness
)

depth = as_1d_floats("depth", depth, n=n_patches, unit="meters")
check_range("dip", dip, 0.0, 90.0, unit="degrees")
```

## Interactive `validate()` reports

`Fault.validate()`, `DataSet.validate()` (GNSS/InSAR/Vertical), and
`Mesh.validate()` check for setups that are *legal but physically suspect*
and return a `ValidationReport` of `ValidationIssue` entries:

```python
report = fault.validate()
print(report)            # "ValidationReport: 1 error(s), 1 warning(s)" + lines
report.ok                # False if any error-severity issue
report.issues            # tuple of ValidationIssue(severity, field, message)
report.raise_if_errors() # turn errors into an exception in scripts
```

What each object checks:

| Object | Errors | Warnings |
|---|---|---|
| `Fault` | patch material above the free surface; degenerate triangles | extreme patch aspect ratios; strike outside [0, 360) |
| `GNSS` / `Vertical` | — | duplicated station coordinates; extreme uncertainty spreads |
| `InSAR` | — | the above, plus look vectors pointing down (`look_u < 0`), which usually means satellite-to-ground components where GeoDef expects ground-to-satellite |
| `Mesh` | nodes above the free surface; degenerate triangles; bad connectivity | duplicate or unreferenced nodes; sliver triangles (edge ratio > 20) |

Use the reports in notebooks before investing in a long inversion; use
`raise_if_errors()` in scripts and pipelines.
