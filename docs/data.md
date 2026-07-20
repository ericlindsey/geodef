# `geodef.data` — Geodetic data types

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

Use `data.gnss`, `data.horizontal_gnss`, `data.insar`, or `data.vertical` for
ordinary construction. These functions return the existing validated `GNSS`,
`InSAR`, and `Vertical` classes; no parallel data representation is involved.
The class constructors remain useful when adapting older code or implementing
specialized readers.

Constructor and file column order is consistently longitude, then latitude.

## From measurements to an inverse problem

A `DataSet` contains more than measured values. It defines three things needed
by an inversion:

- the observation vector `d` (GNSS components, InSAR line of sight, or
  vertical displacement);
- the covariance `C_d`, which describes uncertainty and correlation; and
- a projection from modeled East/North/Up displacement into the measurement
  space.

GeoDef minimizes covariance-weighted residuals,
`(d - Gm)^T C_d^-1 (d - Gm)`. Consequently, uncertainties must use the same
physical units as the observations, and covariance entries use units squared.
Do not use tiny formal uncertainties merely to force a dataset to dominate:
include realistic measurement and model-error contributions when possible.

---

## GNSS

Three-component (E, N, U) or horizontal-only (E, N) displacement/velocity data.

```python
from geodef import data

# Three-component observations; scalar uncertainties broadcast over stations.
gnss = data.gnss(
    lon=lon, lat=lat,
    east=east, north=north, up=up,
    sigma_east=0.001, sigma_north=0.001, sigma_up=0.002,
    name="campaign_gnss", station_names=station_names,
)

horizontal = data.horizontal_gnss(
    lon=lon, lat=lat,
    east=east, north=north,
    sigma_east=sigma_east, sigma_north=sigma_north,
)
```

For a velocity field, declare the semantics rather than relying on context:

```python
velocity = data.horizontal_gnss(
    lon=lon, lat=lat,
    east=east, north=north,
    sigma_east=sigma_east, sigma_north=sigma_north,
    quantity="velocity", units="mm/yr",
    epoch="2025.0", time_span=("2020.0", "2025.0"),
)
```

The corresponding displacement units are `m`, `cm`, and `mm`; supported
velocity units are `m/s`, `m/yr`, `cm/yr`, and `mm/yr`. GeoDef records but does
not silently convert these values. Joint inversions require consistent
quantity and units.

The lower-level constructors and text readers remain available:

```python
from geodef import GNSS

# Optional per-station East-North correlation (scalar or per-station array)
gnss = GNSS(lon=lon, lat=lat, ve=ve, vn=vn, vu=vu,
            se=se, sn=sn, su=su, rho=0.4)

# Load from file (columns: lon lat uE uN uZ sigE sigN sigZ)
gnss = GNSS.load("stations.dat")
gnss = GNSS.load("stations.dat", components="en")       # horizontal-only

# Save
gnss.save("out.dat")
gnss.to_gmt("out_gmt.dat")   # lon lat uE uN sigE sigN (for psvelo)
```

Pass `rho` to build a block covariance whose per-station East-North pair has
off-diagonal `rho * se * sn` (the Up component stays uncorrelated). `rho` may be
a scalar or a `(n_stations,)` array and is mutually exclusive with an explicit
`covariance=`.

**Properties:** `lat`, `lon`, `obs`, `sigma`, `covariance`, `n_stations`,
`n_obs`, `components` (`'enu'` or `'en'`), `east`, `north`, `up`,
`sigma_east`, `sigma_north`, `sigma_up`

---

## InSAR

Line-of-sight displacement with per-pixel look vectors.

```python
from geodef import data

insar = data.insar(
    lon=lon, lat=lat, los=los, sigma=sigma,
    look_e=look_e, look_n=look_n, look_u=look_u,
    name="ascending",
)

# Load from file (columns: lon lat uLOS sigLOS losE losN losU)
insar = data.InSAR.load("ascending.dat")

insar.save("out.dat")
insar.to_gmt("out_gmt.dat")   # lon lat uLOS
```

**Properties:** `lat`, `lon`, `obs` (= LOS), `sigma`, `covariance`, `n_stations`, `n_obs`

The look vector is a unit vector in `[East, North, Up]`; GeoDef computes
`u_LOS = look_e*u_e + look_n*u_n + look_u*u_u`. Sign conventions differ among
InSAR products, so verify the supplied vector by projecting a known upward or
eastward displacement before interpreting positive LOS motion.

---

## Vertical

Single-component vertical displacement (coral uplift, tide gauges, etc.).

```python
from geodef import data

vert = data.vertical(
    lon=lon, lat=lat,
    displacement=displacement, sigma=sigma,
    name="leveling", station_names=benchmark_names,
)

# Load from file (columns: lon lat uZ sigZ)
vert = data.Vertical.load("coral.dat")

vert.save("out.dat")
vert.to_gmt("out_gmt.dat")   # lon lat uZ
```

**Properties:** `lat`, `lon`, `obs`, `sigma`, `covariance`, `n_stations`, `n_obs`

---

## Common DataSet interface

All data classes share:

| Attribute/Method | Description |
|-----------------|-------------|
| `data.obs` | Observation vector, shape `(n_obs,)` |
| `data.sigma` | 1-sigma uncertainties, shape `(n_obs,)` |
| `data.covariance` | Full covariance matrix, shape `(n_obs, n_obs)`; diagonal from `sigma` by default |
| `data.project(ue, un, uz)` | Maps displacement components to observation space |
| `data.n_stations` | Number of physical observation locations |
| `data.n_obs` | Length of the observation vector |
| `data.dataset_name` | Stable dataset identifier used in joint results and plots |
| `data.station_names` | Optional per-station labels, shape `(n_stations,)` |
| `data.quantity` | `"displacement"` or `"velocity"` |
| `data.units` | Declared observation and uncertainty units |
| `data.epoch` | Optional representative epoch label |
| `data.time_span` | Optional `(start, end)` epoch labels |

### Constructing from a table

`data.from_table` accepts a plain column mapping, a structured NumPy array, or
an object implementing the Python dataframe interchange protocol. This keeps
Pandas, Polars, and other dataframe libraries optional. Column mappings are
always explicit, so a source's abbreviations and unit-bearing names cannot be
silently misinterpreted.

```python
vertical = data.from_table(
    table,
    kind="vertical",
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

Choose `kind="gnss"`, `"horizontal_gnss"`, `"insar"`, or `"vertical"`.
By default, a missing value raises an error naming both the GeoDef field and
source column. Pass `missing="drop"` to discard every incomplete row before
the normal dataset validation runs.

### Site names

The construction functions accept `station_names=` for per-station labels and
`name=` for the dataset identifier. The class constructors retain their older
`name=` station-array keyword. Dataset identity, station labels, quantity,
units, epoch, and time span round-trip through the built-in text files.
When present, names round-trip through `save()`/`load()` as a leading
`# names:` comment line, so the numeric data block stays unchanged:

```python
gnss = GNSS(lon=lon, lat=lat, ve=ve, vn=vn, vu=vu, se=se, sn=sn, su=su,
            name=["P001", "P002", "P003"], dataset_name="campaign_gnss")
gnss.save("stations.dat")
GNSS.load("stations.dat").station_names  # array(['P001', 'P002', 'P003'])
```

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

The covariance is used automatically by `geodef.invert.solve()` and
`geodef.greens.stack_weights()`.

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
