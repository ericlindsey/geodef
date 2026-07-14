# `geodef.slip` — Slip and displacement values

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

`SlipModel` is GeoDef's canonical named representation of per-patch slip.
NumPy vectors remain accepted by the low-level and legacy APIs, but the named
object records which coordinates the vector actually uses.

## Strike/dip slip

```python
import geodef
import numpy as np

slip = geodef.SlipModel(
    strike=[1.0, 1.2, 0.8],
    dip=[0.2, 0.3, 0.1],
)

slip.strike       # physical strike-slip, shape (N,)
slip.dip          # physical dip-slip, shape (N,)
slip.magnitude    # hypot(strike, dip), shape (N,)
slip.rake         # atan2(dip, strike), degrees
slip.vector       # blocked [strike | dip], shape (2N,)
```

All stored arrays are copied and read-only. `fault.displacement`,
`fault.moment`, `fault.magnitude`, inversion smoothing targets, and slip plots
accept a `SlipModel`; the corresponding NumPy forms remain supported.

## One-component directions

A fixed direction has one independent amplitude per patch. It is not silently
expanded into a two-component model vector:

```python
fixed_rake = geodef.SlipModel.from_rake(amplitude, rake=90.0)
fixed_azimuth = geodef.SlipModel.from_azimuth(
    amplitude,
    azimuth=15.0,
    fault_strike=fault.strike,
)

fixed_rake.n_components   # 1
fixed_rake.vector         # signed amplitudes, shape (N,)
fixed_rake.strike         # derived physical strike-slip
fixed_rake.dip            # derived physical dip-slip
```

`from_strike` and `from_dip` construct the corresponding one-component local
bases. Signed negative amplitudes reverse the physical slip direction;
`magnitude` is always non-negative.

## Plate-motion coordinates

On triangular or curved faults, adjacent patch strike/dip axes can change
sharply even when the large-scale tectonic direction is smooth. Plate
coordinates store two components relative to that larger-scale direction:

```python
plate_rake = np.full(fault.n_patches, 90.0)
slip = geodef.SlipModel.from_plate_rake(
    parallel,
    perpendicular,
    plate_rake=plate_rake,
)

slip.rake_parallel       # first model block
slip.rake_perpendicular  # second model block
slip.vector              # blocked [parallel | perpendicular]
slip.strike              # derived patch-local physical component
slip.dip                 # derived patch-local physical component
```

The rotation follows

```text
strike = parallel*cos(r) - perpendicular*sin(r)
dip    = parallel*sin(r) + perpendicular*cos(r)
```

where `r` is `plate_rake`. This basis belongs to the kinematic slip model, not
`TriGeometry`: the same mesh can be used with different plate-motion
hypotheses.

An Euler pole can supply the smooth geographic direction:

```python
pole = (pole_lat, pole_lon, rate_deg_per_myr)
plate_rake = geodef.plate_rake_from_euler(fault, pole)

result = geodef.invert(
    fault,
    data,
    components="plate",
    plate_rake=plate_rake,
    smoothing="laplacian",
    bounds=(np.array([0.0, -0.1]), np.array([1.0, 0.1])),
)

result.slip_model.rake_parallel
result.slip_model.rake_perpendicular
```

The Green's matrix is rotated into `[parallel | perpendicular]` before the
solve. Smoothing, targets, bounds, covariance, and resolution therefore all
refer to the smooth plate coordinates, while forward deformation and moment
use the derived physical strike/dip components.

`SlipModel.from_euler_pole(parallel, perpendicular, fault=fault, pole=pole)`
constructs a complete model in one call.

## Named displacement

`fault.displacement` returns `Displacement(east, north, up)`:

```python
displacement = fault.displacement(obs_lat, obs_lon, slip)
displacement.east
displacement.north
displacement.up
displacement.vector  # observation-interleaved [E, N, U, E, N, U, ...]

# Existing tuple idiom remains valid
east, north, up = displacement
```

## Patch grid ordering

Structured faults provide explicit conversion helpers, so callers do not need
to memorize which flat axis varies fastest:

```python
grid = fault.reshape_patches(slip.magnitude)
# shape (n_width, n_length): [dip_index, strike_index]

values = fault.flatten_patches(grid)
# shape (n_patches,)
```

Trailing dimensions are preserved, so an `(N, k)` patch-first array becomes
`(n_width, n_length, k)`.
