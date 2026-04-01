# GeoDef

A Python library for forward and inverse modeling of fault slip in elastic half-spaces. Targets coseismic (earthquake) and interseismic (coupling) applications.

## Install

```bash
uv pip install -e .
```

## Quick start

```python
import numpy as np
import geodef
from geodef import Fault, GNSS

# Create a fault (100 km x 50 km, 15° dip, 10×5 patches)
fault = Fault.planar(lat=0.0, lon=100.0, depth=30_000.0,
                     strike=90.0, dip=15.0,
                     length=100_000.0, width=50_000.0,
                     n_length=10, n_width=5)

# Forward model: 1 m dip slip, displacement at 3 stations
obs_lat = np.array([0.1, 0.2, 0.3])
obs_lon = np.array([100.0, 100.1, 100.2])
ue, un, uz = fault.displacement(obs_lat, obs_lon, slip_strike=0.0, slip_dip=1.0)
```

## Inversion

```python
# Load data and invert for slip
gnss = GNSS.load("stations.dat")
insar = geodef.InSAR.load("ascending.dat")

result = geodef.invert(fault, [gnss, insar],
                       smoothing='laplacian',
                       smoothing_strength=1e3,
                       bounds=(0, None))

print(f"Mw = {result.Mw:.2f}, reduced chi2 = {result.chi2:.2f}")
geodef.plot.slip(fault, result.slip_vector)
```

## Examples

| Notebook | What it covers |
|----------|---------------|
| `examples/01_forward_model.ipynb` | Fault creation, Green's matrix, GNSS + InSAR forward prediction |
| `examples/02_caching.ipynb` | Hash-based caching for fast reuse of Green's matrices |
| `examples/03_plotting.ipynb` | All plot types: slip, vectors, InSAR, fit, fault3d, map, resolution |
| `examples/04_mesh_generation.ipynb` | Triangular mesh creation from trace, polygon, slab2.0 grids |

## Module reference

Full API docs with examples are in `docs/`:

| Doc | Module |
|-----|--------|
| [`docs/fault.md`](docs/fault.md) | `Fault` class — factory methods, forward modeling, I/O |
| [`docs/data.md`](docs/data.md) | `GNSS`, `InSAR`, `Vertical` data types |
| [`docs/greens.md`](docs/greens.md) | Green's matrix assembly and Laplacian operators |
| [`docs/invert.md`](docs/invert.md) | Inversion, regularization, hyperparameter tuning, model assessment |
| [`docs/plot.md`](docs/plot.md) | All plot functions |
| [`docs/mesh.md`](docs/mesh.md) | Triangular mesh generation |
| [`docs/okada.md`](docs/okada.md) | `okada` dispatcher + `okada85` / `okada92` direct access |
| [`docs/cache.md`](docs/cache.md) | Disk caching configuration |
| [`docs/transforms.md`](docs/transforms.md) | Geodetic coordinate transforms |

## Testing

```bash
uv run pytest   # 669 tests
```

## References

- Okada (1985), *BSSA* 75(4), 1135–1154.
- Okada (1992), *BSSA* 82(2), 1018–1040.
- Nikkhoo & Walter (2015), *GJI* 201(2), 1119–1141.
- Fukuda & Johnson (2008), *BSSA* 98(3), 1128–1146.
- Lindsey et al. (2021), *Nature Geoscience* 14, 801–807.
