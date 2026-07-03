# GeoDef

A Python library for forward and inverse modeling of fault slip in elastic
half-spaces. Targets coseismic (earthquake) and interseismic (coupling)
applications.

Status: **v1.0** — the runtime library, the ten-part tutorial course, and the
per-module documentation are complete. `ruff` and `mypy` pass cleanly and the
test suite runs warning-free.

## Install

```bash
uv pip install -e .

# optional extras
uv pip install -e ".[geo]"    # pyproj geodetic transforms / slab2.0 sampling
uv pip install -e ".[mesh]"   # meshpy triangular mesh generation
uv pip install -e ".[maps]"   # cartopy geographic map plotting
uv pip install -e ".[all]"    # everything optional
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

# Optional fixed slip directions
fixed_rake = geodef.invert(fault, gnss, components='rake', rake=90.0)
fixed_azimuth = geodef.invert(fault, gnss,
                              components='azimuth', slip_azimuth=15.0)
```

## Tutorials

A ten-part course in geodetic inverse methods, taught with synthetic data and
executed by the pytest suite so it stays aligned with the runtime API:

1. Forward model `d = G m` · 2. Discretization and the `G` matrix ·
3. Unregularized inversion and overfitting · 4. Regularization ·
5. Choosing the regularization strength (L-curve / ABIC / CV) ·
6. Joint GNSS + InSAR · 7. Correlated InSAR noise ·
8. Bounds and constraints · 9. Uncertainty and resolution ·
10. Nonlinear geometry search.

See [`tutorials/README.md`](tutorials/README.md) for the full path.
`tutorials/reference_plots.ipynb` is an exhaustive gallery of the plot functions.

## Examples

Project and real-data examples live in `examples/`.

| Notebook | What it covers |
|----------|---------------|
| `examples/gorkha_earthquake/model_gorkha.ipynb` | Real-data Gorkha earthquake inversion with GNSS, InSAR, smoothing, and fixed-azimuth slip |
| `examples/mesh_generation.ipynb` | Building triangular fault meshes from traces, polygons, points, and slab2.0 |

## Module reference

Full API docs with examples are in `docs/`:

| Doc | Module |
|-----|--------|
| [`docs/fault.md`](docs/fault.md) | `Fault` class — factory methods, forward modeling, I/O |
| [`docs/data.md`](docs/data.md) | `GNSS`, `InSAR`, `Vertical` data types |
| [`docs/greens.md`](docs/greens.md) | Green's matrix assembly and Laplacian operators |
| [`docs/invert.md`](docs/invert.md) | Inversion, regularization, hyperparameter tuning, model assessment |
| [`docs/plot.md`](docs/plot.md) | All plot functions (patches, interpolated slip, vectors, InSAR, 3-D, fit, resolution, uncertainty) |
| [`docs/geomap.md`](docs/geomap.md) | Optional Cartopy geographic map plotting |
| [`docs/mesh.md`](docs/mesh.md) | Triangular mesh generation |
| [`docs/euler.md`](docs/euler.md) | Euler pole fitting and rigid-block velocities |
| [`docs/okada.md`](docs/okada.md) | `okada` dispatcher + `okada85` / `okada92` direct access |
| [`docs/cache.md`](docs/cache.md) | Disk caching configuration |
| [`docs/transforms.md`](docs/transforms.md) | Geodetic coordinate transforms |

## Testing

```bash
uv run pytest -q   # 883 passed, 1 skipped, 884 collected
```

The tutorial notebooks and a Gorkha example smoke test run as part of the suite.
A handful of `Fault.load` tests need reference data under `related/stress-shadows/`
and are skipped when it is absent.

## AI co-authorship

All code in this repository has been co-authored with Claude Opus 4.6, Claude
Opus 4.8, and Codex 5.5. Keep this model list current when future AI models make
material contributions.

## References

- Okada (1985), *BSSA* 75(4), 1135–1154.
- Okada (1992), *BSSA* 82(2), 1018–1040.
- Nikkhoo & Walter (2015), *GJI* 201(2), 1119–1141.
- Fukuda & Johnson (2008), *BSSA* 98(3), 1128–1146.
- Lindsey et al. (2021), *Nature Geoscience* 14, 801–807.
