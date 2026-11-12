# `geodef.data` — Geodetic data types

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

A dataset pairs observation locations with measured values and their
uncertainties. There are three kinds, all importable from the top level:
`geodef.GNSS`, `geodef.InSAR`, and `geodef.Vertical`. Each has **one**
constructor with keyword arguments; coordinates are always `lon=` then `lat=`,
and any value or uncertainty may be a scalar, which is repeated for every
station.

## From measurements to an inverse problem

A `DataSet` contains more than measured values. It defines three things needed
by an inversion:

- the observation vector `d` (`data.obs`);
- the covariance `C_d` (`data.covariance`), which describes uncertainty and
  correlation; and
- a projection from modeled East/North/Up displacement into the measurement
  space (`data.project(east, north, up)`).

GeoDef minimizes covariance-weighted residuals,
`(d - Gm)^T C_d^-1 (d - Gm)`. Consequently, uncertainties must use the same
physical units as the observations, and covariance entries use units squared.
Do not use tiny formal uncertainties merely to force a dataset to dominate:
include realistic measurement and model-error contributions when possible.

---

## GNSS

East, North, and (optionally) Up displacements or velocities at stations.

```python
import geodef

# One uncertainty for every component
gnss = geodef.GNSS(lon=lon, lat=lat, east=east, north=north, up=up, sigma=0.002)

# Component-specific uncertainties, a dataset name, and station labels
gnss = geodef.GNSS(
    lon=lon, lat=lat,
    east=east, north=north, up=up,
    sigma_east=0.001, sigma_north=0.001, sigma_up=0.003,
    name="campaign_gnss", station_names=["P001", "P002", "P003"],
)

# Horizontal-only: leave out `up`
horizontal = geodef.GNSS(lon=lon, lat=lat, east=east, north=north, sigma=0.001)
```

`sigma=` is the default for any component without its own `sigma_east`,
`sigma_north`, or `sigma_up`.

For a velocity field, declare the semantics rather than relying on context:

```python
velocity = geodef.GNSS(
    lon=lon, lat=lat, east=east, north=north, sigma=0.5,
    quantity="velocity", units="mm/yr",
    epoch="2025.0", time_span=("2020.0", "2025.0"),
)
```

The displacement units are `m`, `cm`, and `mm`; the velocity units are `m/s`,
`m/yr`, `cm/yr`, and `mm/yr`. GeoDef records but does not convert these values.
Joint inversions require consistent quantity and units.

Pass `rho` to build a covariance whose per-station East-North pair has
off-diagonal `rho * sigma_east * sigma_north` (the Up component stays
uncorrelated). `rho` may be a scalar or a `(n_stations,)` array and is mutually
exclusive with an explicit `covariance=`.

```python
gnss = geodef.GNSS(lon=lon, lat=lat, east=east, north=north, sigma=0.001, rho=0.4)

# Files (columns: lon lat east north up sigma_east sigma_north sigma_up)
gnss = geodef.GNSS.load("stations.dat")
gnss = geodef.GNSS.load("stations.dat", components="en")   # horizontal-only
gnss.save("out.dat")
gnss.to_gmt("out_gmt.dat")   # lon lat uE uN sigE sigN (for psvelo)
```

**Properties:** `east`, `north`, `up`, `sigma_east`, `sigma_north`, `sigma_up`,
`components` (`'enu'` or `'en'`), plus the common interface below.

---

## InSAR

Line-of-sight values with per-pixel look vectors.

```python
insar = geodef.InSAR(
    lon=lon, lat=lat, los=los, sigma=0.005,
    look_e=look_e, look_n=look_n, look_u=look_u,
    name="ascending",
)

# Files (columns: lon lat los sigma look_e look_n look_u)
insar = geodef.InSAR.load("ascending.dat")
insar.save("out.dat")
insar.to_gmt("out_gmt.dat")   # lon lat uLOS
```

The look vector is a **unit** vector in `[East, North, Up]` pointing from the
ground to the satellite; GeoDef computes
`u_LOS = look_e*u_e + look_n*u_n + look_u*u_u`, so uplift is positive LOS
motion. Pass `normalize_look=True` to renormalize approximate vectors. Sign
conventions differ among InSAR products: `insar.validate()` warns when the
vectors appear to point from the satellite to the ground.

**Properties:** `los`, `look_e`, `look_n`, `look_u`, plus the common interface.

---

## Vertical

Single-component vertical observations (leveling, coral uplift, tide gauges).

```python
vertical = geodef.Vertical(
    lon=lon, lat=lat, displacement=uplift, sigma=0.01,
    name="leveling", station_names=benchmark_names,
)

# Files (columns: lon lat displacement sigma)
vertical = geodef.Vertical.load("coral.dat")
vertical.save("out.dat")
vertical.to_gmt("out_gmt.dat")   # lon lat uZ
```

**Properties:** `displacement`, plus the common interface.

---

## Common DataSet interface

Every dataset class shares:

| Attribute/Method | Description |
|-----------------|-------------|
| `data.lon`, `data.lat` | Station coordinates in degrees, shape `(n_stations,)` |
| `data.obs` | Observation vector, shape `(n_obs,)` |
| `data.sigma` | One-sigma uncertainties, shape `(n_obs,)` |
| `data.covariance` | Covariance matrix, shape `(n_obs, n_obs)`; `diag(sigma**2)` by default |
| `data.project(east, north, up)` | Maps a displacement field at the stations into the layout of `obs` |
| `data.with_obs(values)` | A copy with new observation values (same stations and uncertainties) |
| `data.n_stations`, `data.n_obs` | Number of locations; length of the observation vector |
| `data.name` | Dataset name used to label joint results and plots |
| `data.station_names` | Optional per-station labels, shape `(n_stations,)` |
| `data.quantity`, `data.units` | `"displacement"` or `"velocity"`, and the declared units |
| `data.epoch`, `data.time_span` | Optional epoch label and `(start, end)` labels |
| `data.validate()` | A `ValidationReport` of suspicious-but-legal settings |
| `data.save(path)`, `Class.load(path)` | Round-trip through a whitespace-delimited text file |

Names, quantity, units, epoch, and time span round-trip through `save()` and
`load()` in `#` comment lines, so the numeric block still reads with
`np.loadtxt`.

### Synthetic data

`synthetic` forward-models slip on a fault into a dataset. Give the fault, the
slip, and the constructor arguments other than the observations; pass `rng`
(a `numpy.random.Generator` or an integer seed) to add noise drawn from the
dataset's own uncertainties or covariance.

```python
gnss = geodef.GNSS.synthetic(
    fault, lon=lon, lat=lat, dip_slip=true_slip, sigma=0.005, rng=0,
)
insar = geodef.InSAR.synthetic(
    fault, lon=pixel_lon, lat=pixel_lat, dip_slip=true_slip,
    look_e=-0.38, look_n=0.09, look_u=0.92, sigma=0.005, rng=rng,
)
horizontal = geodef.GNSS.synthetic(
    fault, lon=lon, lat=lat, strike_slip=1.0, sigma=0.002, components="en",
)
```

Without `rng`, `obs` is exactly the model prediction.

### Constructing from a table

`from_table` reads named columns from a mapping, a structured NumPy array, or
a pandas/polars `DataFrame`. `columns` maps constructor arguments to the
table's column names, so a source's abbreviations cannot be silently
misinterpreted; without it, columns must be named like the arguments.

```python
vertical = geodef.Vertical.from_table(
    table,
    columns={
        "lon": "longitude",
        "lat": "latitude",
        "displacement": "uplift_mm",
        "sigma": "uplift_sigma_mm",
        "station_names": "benchmark",
    },
    units="mm",
    name="leveling",
)
```

Remaining keyword arguments go to the constructor (for example a scalar
`sigma=` when the table has no uncertainty column). A missing value raises an
error naming both the argument and the source column; pass `missing="drop"` to
discard every incomplete row first.

### Setting a full covariance matrix

A common workflow is to compute a full covariance from the noise model and pass
it when constructing the dataset:

```python
# Compute full covariance (e.g. from empirical semivariogram)
Cdata = compute_full_covariance(lat, lon, ...)  # (n_obs, n_obs)
insar = InSAR(lon=lon, lat=lat, los=los, sigma=sigma, look_e=look_e,
              look_n=look_n, look_u=look_u, covariance=Cdata)
```

`load()` reads diagonal-uncertainty files. If a loaded dataset needs a full
covariance, reconstruct it from the source arrays and pass `covariance=`.

The covariance is used automatically by `geodef.solve()` and
`geodef.greens.stack_weights()`, and `synthetic(..., rng=...)` draws its noise
from it.

### Building a spatially-correlated covariance

InSAR noise (atmosphere, orbits) is spatially correlated, so a diagonal `C_d`
underestimates its true structure. `geodef.data.spatial_covariance()` builds a full
`C_d` from an isotropic covariance model whose correlation decays with
great-circle distance:

```python
from geodef import InSAR
from geodef.data import spatial_covariance

# sill = correlated variance (m^2), correlation_length in meters
Cdata = spatial_covariance(
    lon, lat, sill=4e-4, correlation_length=8_000.0,
    model="exponential", nugget=1e-4,
)
insar = InSAR(lon=lon, lat=lat, los=los, sigma=sigma, look_e=look_e,
              look_n=look_n, look_u=look_u, covariance=Cdata)
```

`C_ij = sill * rho(d_ij) + nugget * delta_ij`, with `rho(d) = exp(-d / L)`
(`'exponential'`) or `exp(-(d / L)^2)` (`'gaussian'`). The `nugget` adds
uncorrelated white noise on the diagonal. Applies to one-value-per-station
datasets (`InSAR`, `Vertical`).

The covariance must be symmetric positive definite for whitening and inversion.
Dense covariance scales as `O(n_obs^2)` in memory, so full-resolution InSAR
scenes commonly need downsampling or a specialized covariance approximation.
See [covariance matrices](https://en.wikipedia.org/wiki/Covariance_matrix) and
[semivariograms](https://en.wikipedia.org/wiki/Variogram) for background.

---

## Observation vector layout

- **GNSS (3-comp):** interleaved `[e1, n1, u1, e2, n2, u2, ...]`
- **GNSS (2-comp):** interleaved `[e1, n1, e2, n2, ...]`
- **InSAR:** `[los_1, los_2, ...]` (one value per pixel)
- **Vertical:** `[uz_1, uz_2, ...]`
