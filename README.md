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

Status: **v1.1** — the runtime library, the eleven-part tutorial course, the
per-module documentation, and the optional JAX accelerator (differentiable
forward models, gradient-based and Bayesian geometry inference) are complete.
`ruff` and `mypy` pass cleanly and the test suite runs warning-free.

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

print(f"Mw = {result.Mw:.2f}, reduced chi2 = {result.reduced_chi2:.2f}")
geodef.plot.slip(fault, result.slip_vector)

# Optional fixed slip directions
fixed_rake = geodef.invert(fault, gnss, components='rake', rake=90.0)
fixed_azimuth = geodef.invert(fault, gnss,
                              components='azimuth', slip_azimuth=15.0)
```

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
| [`docs/fault.md`](docs/fault.md) | `Fault` class — factory methods, forward modeling, I/O |
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
