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

The planned research-grade coupling example and earthquake example with
nuisance parameters plus operator-scale correlated noise remain deferred until
the corresponding Priority 4 APIs land. Tutorial 12 teaches the present
two-step backslip workflow without pretending those research interfaces exist.
