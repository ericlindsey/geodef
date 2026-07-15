# Slip vectors and basis conversions

> Conventions — axes, depth sign, angles, units, array ordering,
> regularization: see [`conventions.md`](conventions.md).

GeoDef represents slip with NumPy arrays and provides functions for the common
conversions. For `N` patches, a two-component model vector is blocked:
`[strike_slip_0, ..., strike_slip_N, dip_slip_0, ..., dip_slip_N]`.

## Pack and unpack

```python
from geodef import slip

model = slip.pack(strike_slip, dip_slip)
strike_slip, dip_slip = slip.unpack(model)

slip_magnitude = slip.magnitude(strike_slip, dip_slip)
slip_rake = slip.rake(strike_slip, dip_slip)
```

`fault.displacement` accepts the two physical components directly and returns
three arrays:

```python
east, north, up = fault.displacement(
    obs_lat,
    obs_lon,
    slip_strike=strike_slip,
    slip_dip=dip_slip,
)
```

## Fixed rake and geographic azimuth

A one-component inversion solves one signed amplitude per patch. Convert an
amplitude to physical strike/dip components with the matching function:

```python
strike_slip, dip_slip = slip.from_rake(amplitude, rake_degrees=90.0)

strike_slip, dip_slip = slip.from_azimuth(
    amplitude,
    azimuth_degrees=15.0,
    fault_strike_degrees=fault.strike,
)
```

`from_rake` uses each patch's local strike/dip axes. `from_azimuth` preserves a
single geographic direction across a curved mesh by accounting for each
patch's strike.

## Plate-motion coordinates

For a curved triangular mesh, smoothing physical strike/dip components can
inherit abrupt changes in patch orientation. A plate basis instead uses a
smooth large-scale direction:

```python
plate_rake = slip.plate_rake_from_euler(
    fault,
    pole=(pole_lat, pole_lon, rate_degrees_per_myr),
)

strike_slip, dip_slip = slip.from_plate(
    parallel,
    perpendicular,
    plate_rake_degrees=plate_rake,
)

parallel, perpendicular = slip.to_plate(
    strike_slip,
    dip_slip,
    plate_rake_degrees=plate_rake,
)
```

Invert directly in those coordinates with `components="plate"` and
`plate_rake=plate_rake`. The result keeps the solved vector blocked as
`[parallel | perpendicular]` and exposes both the solved and physical views:

```python
result = geodef.invert.solve(
    fault,
    datasets,
    components="plate",
    plate_rake=plate_rake,
)

result.rake_parallel
result.rake_perpendicular
result.strike_slip
result.dip_slip
result.slip_magnitude
result.slip_rake
```

## Patch ordering

Structured faults use `(n_width, n_length)` grid shape, with along-strike index
varying fastest. Use the fault helpers instead of manual reshape assumptions:

```python
grid = fault.reshape_patches(per_patch_values)
values = fault.flatten_patches(grid)
index = fault.patch_index(strike_idx, dip_idx)
```
