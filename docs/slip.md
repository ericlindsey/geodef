# Slip vectors and basis conversions

> Conventions — axes, depth sign, angles, units, array ordering,
> regularization: see [`conventions.md`](conventions.md).

Everyday code gives slip as two per-patch arrays, `strike_slip` and `dip_slip`
(meters; left-lateral and reverse are positive), and reads the same two names
back from an inversion result:

```python
east, north, up = fault.displacement(
    lon=obs_lon, lat=obs_lat, strike_slip=strike_slip, dip_slip=dip_slip,
)
result.strike_slip, result.dip_slip, result.slip_magnitude, result.slip_rake
```

This module is for the cases where the **model vector** itself is the subject:
multiplying by a Green's matrix, building a custom regularization target, or
converting between slip bases. For `N` patches, a two-component model vector is
blocked: `[strike_slip_0, ..., strike_slip_N-1, dip_slip_0, ..., dip_slip_N-1]`.

## Pack and unpack

```python
from geodef import slip

model = slip.pack(strike_slip, dip_slip)
strike_slip, dip_slip = slip.unpack(model)

slip_magnitude = slip.magnitude(strike_slip, dip_slip)
slip_rake = slip.rake(strike_slip, dip_slip)
```

The packed vector is what a Green's matrix multiplies:

```python
G = fault.greens_matrix(lon=obs_lon, lat=obs_lat)      # (3 * n_obs, 2 * N)
east, north, up = (G @ slip.pack(strike_slip, dip_slip)).reshape(-1, 3).T
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
result = geodef.solve(
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

Structured faults are stored row by row: row 0 is the shallowest row, each
following row is one step down dip, and patches advance along strike within a
row. Use the fault helpers instead of manual reshape assumptions:

```python
grid = fault.reshape_patches(per_patch_values)   # shape (n_width, n_length)
values = fault.flatten_patches(grid)
index = fault.patch_index(strike_idx, dip_idx)
```
