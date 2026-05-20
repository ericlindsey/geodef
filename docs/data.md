# `geodef.data` — Geodetic data types

Three concrete data classes (`GNSS`, `InSAR`, `Vertical`) all inherit from
`DataSet`. They define how displacements are projected into observation space
and provide common infrastructure for coordinates, uncertainties, and
covariance.

Constructor and file column order is consistently longitude, then latitude.

---

## GNSS

Three-component (E, N, U) or horizontal-only (E, N) displacement/velocity data.

```python
from geodef import GNSS

# Construct directly
gnss = GNSS(lon, lat, ve, vn, vu, se, sn, su)          # full 3-component
gnss = GNSS(lon, lat, ve, vn, None, se, sn, None)      # horizontal-only

# Load from file (columns: lon lat uE uN uZ sigE sigN sigZ)
gnss = GNSS.load("stations.dat")
gnss = GNSS.load("stations.dat", components="en")       # horizontal-only

# Save
gnss.save("out.dat")
gnss.to_gmt("out_gmt.dat")   # lon lat uE uN sigE sigN (for psvelo)
```

**Properties:** `lat`, `lon`, `obs`, `sigma`, `covariance`, `n_stations`,
`n_obs`, `components` (`'enu'` or `'en'`)

---

## InSAR

Line-of-sight displacement with per-pixel look vectors.

```python
from geodef import InSAR

insar = InSAR(lon, lat, los, sigma, look_e, look_n, look_u)

# Load from file (columns: lon lat uLOS sigLOS losE losN losU)
insar = InSAR.load("ascending.dat")

insar.save("out.dat")
insar.to_gmt("out_gmt.dat")   # lon lat uLOS
```

**Properties:** `lat`, `lon`, `obs` (= LOS), `sigma`, `covariance`, `n_stations`, `n_obs`

---

## Vertical

Single-component vertical displacement (coral uplift, tide gauges, etc.).

```python
from geodef import Vertical

vert = Vertical(lon, lat, displacement, sigma)

# Load from file (columns: lon lat uZ sigZ)
vert = Vertical.load("coral.dat")

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

### Setting a full covariance matrix

A common workflow is to compute a full covariance from the noise model and pass
it when constructing the dataset:

```python
# Compute full covariance (e.g. from empirical semivariogram)
Cdata = compute_full_covariance(lat, lon, ...)  # (n_obs, n_obs)
insar = InSAR(lon, lat, los, sigma, look_e, look_n, look_u, covariance=Cdata)
```

`load()` reads diagonal-uncertainty files. If a loaded dataset needs a full
covariance, reconstruct it from the source arrays and pass `covariance=`.

The covariance is used automatically by `geodef.invert()` and
`geodef.stack_weights()`.

---

## Observation vector layout

- **GNSS (3-comp):** interleaved `[e1, n1, u1, e2, n2, u2, ...]`
- **GNSS (2-comp):** interleaved `[e1, n1, e2, n2, ...]`
- **InSAR:** `[los_1, los_2, ...]` (one value per pixel)
- **Vertical:** `[uz_1, uz_2, ...]`
