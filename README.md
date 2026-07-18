# GeoDef

A Python library for forward and inverse modeling of fault slip in elastic
half-spaces. Targets coseismic (earthquake) and interseismic (coupling)
applications.

It ships rectangular (Okada 1985/1992) and triangular (Nikkhoo & Walter 2015)
dislocation engines, Green's-matrix assembly, and a full linear-inversion stack
(regularization, hyperparameter selection, bounds, uncertainty and resolution),
all on a pure-NumPy default path. An **optional JAX backend** JIT-compiles the
Green's-function kernels through XLA — on ordinary CPUs and on GPUs when
available — and makes the whole geometry → `G` → slip → data pipeline
**automatically differentiable**. That unlocks gradient-based nonlinear geometry
inversion (`geodef.gradients`, `invert.geometry_search`) and a collapsed
**Bayesian** geometry sampler (`geodef.bayes`, NUTS via blackjax). NumPy stays
the default everywhere; nothing changes for existing users unless a backend is
explicitly selected.

Status: **v0.1** — the runtime library, the first-generation tutorial course, the
per-module documentation, and the optional JAX accelerator (differentiable
forward models, gradient-based and Bayesian geometry inference) are complete.
`ruff` and `mypy` pass cleanly and the test suite runs warning-free. Version
1.0 is reserved for completion of the roadmap followed by human testing.

## Install

```bash
uv pip install -e .

# optional extras
uv pip install -e ".[geo]"    # pyproj geodetic transforms
uv pip install -e ".[mesh]"   # meshpy + netCDF4 mesh generation / slab2.0
uv pip install -e ".[maps]"   # cartopy geographic map plotting
uv pip install -e ".[jax]"    # JAX backend: JIT/GPU kernels + autodiff
uv pip install -e ".[bayes]"  # Bayesian geometry sampling (jax + blackjax)
uv pip install -e ".[all]"    # everything optional
```

On a machine with an NVIDIA GPU, install JAX's CUDA build instead of the plain
`[jax]` extra (see [`docs/backend.md`](docs/backend.md) for precision and GPU
notes).

### Capabilities at a glance

| Capability | Modules | Requires |
|---|---|---|
| Rectangular dislocations (surface + depth) | `okada85`, `okada92`, `okada` | base install |
| Triangular dislocations | `tri` | base install |
| Fault geometry, forward models, moment | `fault` | base install |
| Elastic medium parameters | `medium` | base install |
| GNSS / InSAR / vertical datasets | `data` | base install |
| Green's assembly, Laplacians, caching | `greens`, `cache` | base install |
| Linear slip inversion + model assessment | `invert` | base install |
| Euler poles and rigid-block velocities | `euler` | base install |
| Input validation and geometry checks | `validation` | base install |
| Slip, data, fit, and 3-D plotting | `plot` | base install |
| High-precision geodetic transforms | `transforms` | `[geo]` |
| Triangular mesh generation, slab2.0 | `mesh` | `[mesh]` |
| Geographic basemap plotting | `geomap` | `[maps]` |
| JIT/GPU kernels, differentiable models | `backend`, `gradients` | `[jax]` |
| Bayesian geometry + slip posteriors | `bayes` | `[bayes]` |

## Five-minute quickstart

Create a fault, generate noisy synthetic GNSS observations, recover slip, and
plot the fit with the base install:

```python
import matplotlib.pyplot as plt
import numpy as np
import geodef

rng = np.random.default_rng(0)
fault = geodef.Fault.planar(
    lat=34.0, lon=-118.0, depth=8_000.0, strike=90.0, dip=30.0,
    length=24_000.0, width=12_000.0, n_length=6, n_width=3,
)
station_lon, station_lat = np.meshgrid(
    np.linspace(-118.18, -117.82, 7), np.linspace(33.88, 34.12, 5)
)
station_lon, station_lat = station_lon.ravel(), station_lat.ravel()
centers = fault.centers_local
true_dip_slip = 1.2 * np.exp(
    -(centers[:, 0] / 7_000.0) ** 2
    - ((centers[:, 1] + 2_000.0) / 5_000.0) ** 2
)
east, north, up = fault.displacement(
    station_lat, station_lon, slip_strike=0.0, slip_dip=true_dip_slip
)
gnss = geodef.data.gnss(
    lon=station_lon, lat=station_lat,
    east=east + rng.normal(0.0, 0.004, east.size),
    north=north + rng.normal(0.0, 0.004, north.size),
    up=up + rng.normal(0.0, 0.008, up.size),
    sigma_east=0.004, sigma_north=0.004, sigma_up=0.008,
    name="synthetic_gnss",
)
result = geodef.solve(
    fault, datasets=gnss, components="dip",
    regularization="laplacian", regularization_strength=1.0,
    bounds=(0.0, None),
)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
geodef.plot.slip(
    fault, result.dip_slip, ax=axes[0], title="Recovered dip slip",
    colorbar_label="Slip (m)",
)
geodef.plot.prediction(result, ax=axes[1])
plt.show()
```

The annotated [quickstart](docs/quickstart.md) explains each step and useful
variations. Use the [workflow and decision guides](docs/workflow.md) to choose
the next method and the [glossary](docs/glossary.md) for notation.

## Differentiable and Bayesian modeling (JAX)

Selecting the JAX backend JIT-compiles the kernels and makes the forward model
differentiable — no change to the calls above, and the teaching notebooks stay
on the NumPy path.

```python
geodef.backend.set_backend("jax")      # pip install geodef[jax]

# Starting geometry [e0, n0, depth, strike, dip, length, width]
theta0 = np.array([0.0, 0.0, 25e3, 90.0, 15.0, 100e3, 50e3])

# Gradient-based nonlinear geometry inversion (variable projection + L-BFGS-B)
gs = geodef.geometry_search(theta0, gnss, ref_lat=0.0, ref_lon=100.0,
                            free=["dip", "depth"])
print(gs.theta, gs.theta_cov)          # best-fit geometry + Gauss-Newton covariance

# Full posterior over geometry + hyperparameters (slip marginalized), via NUTS
from geodef import bayes                # pip install geodef[bayes]
post = bayes.RectPosterior(theta0, gnss, ref_lat=0.0, ref_lon=100.0,
                           free=["dip", "depth"])
result = bayes.sample(post, n_chains=4)
print(result.summary())                # R-hat, ESS, credible intervals
```

See [`docs/backend.md`](docs/backend.md), [`docs/gradients.md`](docs/gradients.md),
and [`docs/bayes.md`](docs/bayes.md) for the full APIs, and
`examples/bayesian_geometry.ipynb` for a worked posterior study.

## Tutorials

An eleven-part course in geodetic inverse methods, taught with synthetic data and
executed by the pytest suite so it stays aligned with the runtime API:

1. Forward model `d = G m` · 2. Discretization and the `G` matrix ·
3. Unregularized inversion and overfitting · 4. Regularization ·
5. Choosing the regularization strength (L-curve / ABIC / CV) ·
6. Joint GNSS + InSAR · 7. Correlated InSAR noise ·
8. Bounds and constraints · 9. Uncertainty and resolution ·
10. Nonlinear geometry search · 11. Gradient-based geometry inversion on the
JAX backend (requires `geodef[jax]`).

Notebooks 1–10 run on the NumPy default path; notebook 11 is an advanced JAX
extension. See [`tutorials/README.md`](tutorials/README.md) for the full path.
`tutorials/reference_plots.ipynb` is an exhaustive gallery of the plot functions.

## Examples

Project and real-data examples live in `examples/`.

| Notebook | What it covers |
|----------|---------------|
| `examples/gorkha_earthquake/model_gorkha.ipynb` | Real-data Gorkha earthquake inversion with GNSS, InSAR, smoothing, and fixed-azimuth slip |
| `examples/mesh_generation.ipynb` | Building triangular fault meshes from traces, polygons, points, and slab2.0 |
| `examples/bayesian_geometry.ipynb` | Collapsed Bayesian geometry inference: NUTS posterior vs Gauss-Newton, slip credible intervals, and an emcee cross-check |

See [`examples/README.md`](examples/README.md) for the full list.

## Module reference

Full API docs with examples are in `docs/`:

| Doc | Module |
|-----|--------|
| [`docs/quickstart.md`](docs/quickstart.md) | Complete first forward model and inversion |
| [`docs/workflow.md`](docs/workflow.md) | API-level map and scientific decision guides |
| [`docs/glossary.md`](docs/glossary.md) | Geophysical and inverse-theory notation |
| [`docs/fault.md`](docs/fault.md) | `Fault` class — factory methods, forward modeling, I/O |
| [`docs/slip.md`](docs/slip.md) | Slip-vector functions, plate-motion coordinates, and patch ordering |
| [`docs/medium.md`](docs/medium.md) | `ElasticMedium` half-space parameters |
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
| [`docs/validation.md`](docs/validation.md) | Input validation helpers and `.validate()` reports |
| [`docs/backend.md`](docs/backend.md) | JAX backend selection, precision, and GPU notes |
| [`docs/gradients.md`](docs/gradients.md) | Differentiable forward models and Jacobians (JAX) |
| [`docs/bayes.md`](docs/bayes.md) | Collapsed Bayesian geometry inference (NUTS / blackjax) |

## Testing

```bash
uv run pytest -q
```

The tutorial notebooks and a Gorkha example smoke test run as part of the suite.
Tests are skipped rather than failed when their optional dependency is absent:
the JAX/blackjax-gated backend, gradient, and Bayesian tests skip without
`geodef[jax]` / `geodef[bayes]`, and a handful of `Fault.load` tests need
reference data under `related/stress-shadows/`.

## Development

Contributor and roadmap docs live at the repository root:

- [`PYTHON.md`](PYTHON.md) — mandatory coding standards and tooling (read before editing any code).
- [`PLAN.md`](PLAN.md) — the forward-looking roadmap (GPU/autodiff, earthquake-cycle modeling, more Green's engines).
- [`CHANGELOG.md`](CHANGELOG.md) — notable changes per release.
- [`COMPATIBILITY.md`](COMPATIBILITY.md) — public-API, versioning, and deprecation policy.
- [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md) — agent onboarding guides for automated contributors.

## AI co-authorship

All code in this repository has been co-authored with Claude Opus 4.6, Claude
Opus 4.8, Claude Fable 5, and Codex 5.5. Keep this model list current when
future AI models make material contributions.

## License and citation

GeoDef is released under the [MIT License](LICENSE). If you use it in
published work, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff) along with the original method papers below.

## References

- Okada (1985), *BSSA* 75(4), 1135–1154.
- Okada (1992), *BSSA* 82(2), 1018–1040.
- Nikkhoo & Walter (2015), *GJI* 201(2), 1119–1141.
- Fukuda & Johnson (2008), *BSSA* 98(3), 1128–1146.
- Lindsey et al. (2021), *Nature Geoscience* 14, 801–807.
