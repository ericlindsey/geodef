# GeoDef Examples

This directory is for project-style and real-data examples. Introductory,
synthetic teaching notebooks live in `tutorials/` and are executed by pytest.

Every example states its scientific question, data provenance, assumptions,
preprocessing, environment/seed contract, validation, and interpretation. A
bundled fixture is identified separately from a downloaded or raw source.

| Example | What it covers |
|---------|----------------|
| `gorkha_earthquake/model_gorkha.ipynb` | Real-data Gorkha earthquake inversion with GNSS, InSAR, regularization, and fixed-azimuth slip |
| `gorkha_earthquake/setup_data.ipynb` | One-time conversion of the bundled Gorkha source data into GeoDef input files |
| `mesh_generation.ipynb` | Building triangular fault meshes from traces, polygons, points, and slab2.0 grids |
| `bayesian_geometry.ipynb` | Collapsed Bayesian geometry inference: NUTS posterior vs Gauss-Newton, slip credible intervals, weak-prior mode, and an emcee cross-check |

## Testing

The synthetic Bayesian and mesh examples use explicit or deterministic inputs.
The Gorkha model has about 2,800 patches, so full Green's assembly remains a
manual path. A reduced smoke test loads its bundled fault and datasets through
the public API to catch format or API drift without running the full inversion.

`mesh_generation.ipynb` needs the `geodef[mesh]` extra and, for its slab2.0
section, a Slab2 depth grid downloaded into `notebook_data/` (the notebook
names the file). `bayesian_geometry.ipynb` needs `geodef[bayes]` plus `emcee`
for its final cross-check. Neither is executed by the test suite.

A research-grade coupling example and an earthquake example with nuisance
parameters and operator-scale correlated noise wait on roadmap items F2-F4
(`plans/FEATURES.md`). Tutorial 12 teaches the present two-step backslip
workflow.
