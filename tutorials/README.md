# GeoDef tutorial course

This fifteen-chapter course teaches geodetic inverse methods through GeoDef.
The markdown is the lesson: theory is developed before the code, each figure is
interpreted, and every chapter ends with checkpoints, common mistakes, a recap,
exercises, and annotated reading. Code stays deliberately small and synthetic
experiments use fixed seeds.

Start with the [five-minute quickstart](../docs/quickstart.md) if you want a
working result immediately. Use the [workflow map](../docs/workflow.md) to
choose a method and the [glossary](../docs/glossary.md) for course notation.

## Course map

Chapter 00 begins with notebook cells and variables, so students with no prior
programming experience can start there. Readers already comfortable with basic
Python, NumPy, matplotlib, and matrix calculations may skip it. Chapters 01–10
form the main sequence. Chapters 11–14 are topic branches and can be read after
their listed prerequisites.

| # | Notebook | Focus | Time | Requires |
|---:|---|---|---:|---|
| 00 | `00_preflight.ipynb` | Python basics, NumPy arrays, plotting, matrix calculations | 3–4 hr | base |
| 01 | `01_forward_model.ipynb` | Elastic dislocations, geometry, `d = Gm`, moment | 60–90 min | base |
| 02 | `02_discretization_and_g_matrix.ipynb` | Basis functions, Green's matrix, ordering, conditioning, cache trust | 60–90 min | base |
| 03 | `03_unregularized_inversion.ipynb` | WLS derivation, SVD, reduced chi-squared, overfitting | 60–90 min | base |
| 04 | `04_regularization.ipynb` | Operators, filter factors, L-curve, ABIC, cross-validation | 90–120 min | base |
| 05 | `05_multiple_datasets.ipynb` | Named GNSS/InSAR construction, LOS, stacking, relative weighting | 60–90 min | base |
| 06 | `06_correlated_noise.ipynb` | Spatial covariance, whitening, effective observations | 45–75 min | base |
| 07 | `07_bounds_and_constraints.ipynb` | NNLS, bounds, inequalities, rake/azimuth/plate bases | 60–90 min | base |
| 08 | `08_uncertainty_and_resolution.ipynb` | Covariance, resolution kernels, synthetic recovery, moment error | 60–90 min | base |
| 09 | `09_nonlinear_geometry.ipynb` | Variable projection, trade-offs, scans, local refinement | about 60 min | base |
| 10 | `10_gradient_geometry.ipynb` | Autodiff geometry search and Gauss–Newton uncertainty | about 60 min | `geodef[jax]` |
| 11 | `11_triangular_faults.ipynb` | Unstructured meshes and direction bases | 30–60 min | base; Chapter 02 |
| 12 | `12_interseismic_coupling.ipynb` | Reference frames, backslip, coupling, moment deficit | 45–60 min | base; Chapter 07 |
| 13 | `13_model_misspecification.ipynb` | Sensitivity analysis, bias, residual forensics | 45–60 min | base; Chapters 08–09 |
| 14 | `14_bayesian_inversion.ipynb` | Priors, positivity, NUTS, sensitivity, convergence, prediction | 90–120 min | `geodef[bayes]`; Chapter 08 |

The first two double-demonstrations deliberately expose the blocked array
layout. Routine later chapters use named result views and named per-dataset
diagnostics so ordering is manipulated only when ordering is the lesson.

## Reference material

- `reference_plots.ipynb` — an exhaustive gallery of every `geodef.plot`
  function, kept outside the numbered methods path (not executed in CI).
- Real mesh building (traces, polygons, slab2.0) lives as a worked example in
  `examples/mesh_generation.ipynb`.

## Worked solutions

One spoiler-separated notebook per chapter lives in `solutions/`. Each
exercise is restated, solved, and interpreted; challenge solutions discuss
alternatives where useful.

| Chapters | Solution notebooks |
|---|---|
| 00–04 | `00_preflight_solutions.ipynb` through `04_regularization_solutions.ipynb` |
| 05–09 | `05_multiple_datasets_solutions.ipynb` through `09_nonlinear_geometry_solutions.ipynb` |
| 10–14 | `10_gradient_geometry_solutions.ipynb` through `14_bayesian_inversion_solutions.ipynb` |

## Running the execution checks

The tutorial notebooks are designed to execute from their own directory with
the repository `src/` path available and matplotlib's noninteractive backend.
The current pytest harness also retains compatibility paths for the
first-generation numbering while downstream test work can migrate atomically:

```bash
uv run pytest tests/test_tutorials.py -q
```
