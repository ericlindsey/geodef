# `geodef.data` — Geodetic data types

Three concrete data classes (`GNSS`, `InSAR`, `Vertical`) all inherit from `DataSet`. They define how displacements are projected into observation space and provide common infrastructure for coordinates, uncertainties, and covariance.

---

## GNSS

Three-component (E, N, U) or horizontal-only (E, N) displacement/velocity data.

```python
from geodef import GNSS

# Construct directly
gnss = GNSS(lat, lon, ve, vn, vu, se, sn, su)          # full 3-component
gnss = GNSS(lat, lon, ve, vn, None, se, sn, None)       # horizontal-only

# Load from file (columns: lon lat hgt uE uN uZ sigE sigN sigZ)
gnss = GNSS.load("stations.dat")
gnss = GNSS.load("stations.dat", components="en")       # horizontal-only

# Save
gnss.save("out.dat")
gnss.to_gmt("out_gmt.dat")   # lon lat uE uN sigE sigN (for psvelo)
```

**Properties:** `lat`, `lon`, `ve`, `vn`, `vu`, `se`, `sn`, `su`, `obs`, `sigma`, `covariance`, `n_stations`, `n_obs`, `components` (`'enu'` or `'en'`)

---

## InSAR

Line-of-sight displacement with per-pixel look vectors.

```python
from geodef import InSAR

insar = InSAR(lat, lon, los, sigma, look_e, look_n, look_u)

# Load from file (columns: lon lat hgt uLOS sigLOS losE losN losU)
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

vert = Vertical(lat, lon, displacement, sigma)

# Load from file (columns: lon lat hgt uZ sigZ)
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

A common workflow is to load data first, compute a full covariance from the noise model, then reconstruct with the covariance:

```python
insar = InSAR.load("ascending.dat")

# Compute full covariance (e.g. from empirical semivariogram)
Cdata = compute_full_covariance(insar.lat, insar.lon, ...)  # (n_obs, n_obs)

# Reconstruct with covariance
insar = InSAR(insar.lat, insar.lon, insar.obs, insar.sigma,
              insar._look_e, insar._look_n, insar._look_u,
              covariance=Cdata)
```

Or pass `covariance=` directly at construction time if building from arrays:

```python
insar = InSAR(lat, lon, los, sigma, look_e, look_n, look_u, covariance=Cdata)
```

The covariance is used automatically by `geodef.invert()` and `stack_weights()`.

---

## Observation vector layout

- **GNSS (3-comp):** interleaved `[e1, n1, u1, e2, n2, u2, ...]`
- **GNSS (2-comp):** interleaved `[e1, n1, e2, n2, ...]`
- **InSAR:** `[los_1, los_2, ...]` (one value per pixel)
- **Vertical:** `[uz_1, uz_2, ...]`
