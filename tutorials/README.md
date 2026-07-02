# GeoDef Tutorials

A progressive, scaffolded path for learning fault-slip forward and inverse
modeling with GeoDef, using synthetic data throughout. Each notebook follows the
same teaching structure — *learning objectives → concepts (with equations) →
worked code → exercises → checkpoint questions → common mistakes → summary* — and
keeps the code deliberately short by leaning on the GeoDef API.

Where it genuinely aids understanding, a notebook uses a **double-demo**: one
cell spells out the underlying calculation by hand, immediately followed by the
one-line GeoDef equivalent, with an assertion that the two agree. This shows
exactly what the library does for you without turning it into a black box.

## Available notebooks

A ten-part course in geodetic inverse methods. Work through them in order; the
recurring synthetic scenario introduced in notebook 03 is reused through 09.

| Notebook | What it covers |
|----------|----------------|
| `01_forward_model.ipynb` | Fault creation, the linear forward model `d = G m`, predicting displacements (`G @ m` vs. `fault.displacement()`), moment magnitude |
| `02_discretization_and_g_matrix.ipynb` | Discretization, `G` as a design matrix, building `G` column by column vs. `greens_matrix()`, blocked-column / interleaved-row layout |
| `03_unregularized_inversion.ipynb` | The linear inverse problem, (weighted) least squares, ill-conditioning, and the overfitting catastrophe |
| `04_regularization.ipynb` | Tikhonov regularization; smoothing, damping, and stress-kernel operators; the effect of the strength `λ` |
| `05_choosing_regularization.ipynb` | Selecting `λ` with the L-curve, ABIC, and cross-validation |
| `06_multiple_datasets.ipynb` | Joint GNSS + InSAR inversion, the line-of-sight projection, relative weighting |
| `07_correlated_noise.ipynb` | Spatially-correlated InSAR noise, building a full `C_d` with `spatial_covariance()`, its effect on uncertainty |
| `08_bounds_and_constraints.ipynb` | NNLS, bounded least squares, inequality constraints, fixed-rake bases |
| `09_uncertainty_and_resolution.ipynb` | Posterior covariance, the resolution matrix, checkerboard tests, `M_w` with error bars |
| `10_nonlinear_geometry.ipynb` | Searching for fault geometry: variable projection, grid search, `scipy.optimize`, an MCMC outlook |

## Reference material

- `reference_plots.ipynb` — an exhaustive gallery of every `geodef.plot`
  function, kept outside the numbered methods path (not executed in CI).
- Real mesh building (traces, polygons, slab2.0) lives as a worked example in
  `examples/mesh_generation.ipynb`.

## Running the execution checks

The live tutorial sequence is executed end-to-end by `tests/test_tutorials.py`
as part of the normal pytest suite:

```bash
uv run pytest tests/test_tutorials.py -q
```
