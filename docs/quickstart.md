# GeoDef five-minute quickstart

This complete example creates a small thrust fault, generates reproducible
synthetic GNSS observations, recovers dip slip, and compares observations with
predictions. It uses the base install and avoids manual slip packing and data
slicing.

```python
import matplotlib.pyplot as plt
import numpy as np

import geodef

rng = np.random.default_rng(0)

# Geometry is in meters and degrees; depth is positive down.
fault = geodef.Fault.planar(
    lat=34.0,
    lon=-118.0,
    depth=8_000.0,
    strike=90.0,
    dip=30.0,
    length=24_000.0,
    width=12_000.0,
    n_length=6,
    n_width=3,
)

# A compact GNSS network around the fault.
station_lon, station_lat = np.meshgrid(
    np.linspace(-118.18, -117.82, 7),
    np.linspace(33.88, 34.12, 5),
)
station_lon = station_lon.ravel()
station_lat = station_lat.ravel()

# A smooth synthetic dip-slip distribution and its surface displacement.
centers = fault.centers_local
true_dip_slip = 1.2 * np.exp(
    -(centers[:, 0] / 7_000.0) ** 2
    - ((centers[:, 1] + 2_000.0) / 5_000.0) ** 2
)
east, north, up = fault.displacement(
    station_lat,
    station_lon,
    slip_strike=0.0,
    slip_dip=true_dip_slip,
)

# Add seeded measurement noise and declare the same uncertainties to GeoDef.
sigma_horizontal = 0.004
sigma_vertical = 0.008
gnss = geodef.data.gnss(
    lon=station_lon,
    lat=station_lat,
    east=east + rng.normal(0.0, sigma_horizontal, east.size),
    north=north + rng.normal(0.0, sigma_horizontal, north.size),
    up=up + rng.normal(0.0, sigma_vertical, up.size),
    sigma_east=sigma_horizontal,
    sigma_north=sigma_horizontal,
    sigma_up=sigma_vertical,
    name="synthetic_gnss",
)

# Solve one dip-slip amplitude per patch with smoothness and non-negativity.
result = geodef.solve(
    fault,
    datasets=gnss,
    components="dip",
    regularization="laplacian",
    regularization_strength=1.0,
    bounds=(0.0, None),
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
geodef.plot.slip(
    fault,
    result.dip_slip,
    ax=axes[0],
    title="Recovered dip slip",
    colorbar_label="Slip (m)",
)
geodef.plot.prediction(result, ax=axes[1])
plt.show()
```

The left panel should recover one broad positive-slip patch rather than every
small fluctuation in the noisy data. The right panel should cluster around the
one-to-one line. `result.reduced_chi2` near one means that the residual scale is
consistent with the declared uncertainties; it does not prove that the fault
geometry or regularization assumption is correct.

## What each part means

| Step | Scientific role | Learn it in |
|---|---|---|
| `Fault.planar(...)` | Declare source geometry and discretization | Tutorials 01–02 |
| `fault.displacement(...)` | Evaluate the elastic forward model | Tutorial 01 |
| `data.gnss(...)` | Pair observations with uncertainty and metadata | Tutorials 03 and 05 |
| `solve(...)` | Estimate slip from the observations | Tutorials 03–04 |
| `regularization="laplacian"` | Prefer spatially smooth slip | Tutorial 04 |
| `bounds=(0, None)` | Exclude opposite-sense slip | Tutorial 07 |
| `result.dip_slip` | Read the named physical component | Tutorials 03–04 |
| `plot.prediction(result)` | Inspect the data fit without manual slicing | Tutorials 03 and 05 |

Before adapting the example, read the [conventions](conventions.md), then use
the [workflow and decision guides](workflow.md) to choose geometry, data,
regularization, constraints, and assessment methods. The
[glossary](glossary.md) connects the mathematical notation to API names.

## First useful variations

- Change `dip` or `depth`, regenerate the synthetic observations, and predict
  how the displacement pattern changes before plotting it.
- Change `regularization_strength` by factors of ten. Smaller values follow
  noise more closely; larger values suppress spatial detail.
- Remove `bounds` and look for negative dip-slip patches. A better fit is not
  automatically a more plausible model.
- Replace synthetic arrays with a real `GNSS.load(...)` dataset, but keep units
  consistent and inspect `gnss.validate()` before inversion.
