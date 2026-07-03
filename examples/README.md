# GeoDef Examples

This directory is for project-style and real-data examples. Introductory,
synthetic teaching notebooks live in `tutorials/` and are executed by pytest.

| Example | What it covers |
|---------|----------------|
| `gorkha_earthquake/model_gorkha.ipynb` | Real-data Gorkha earthquake inversion with GNSS, InSAR, smoothing, and fixed-azimuth slip |
| `gorkha_earthquake/setup_data.ipynb` | One-time conversion of the bundled Gorkha source data into GeoDef input files |
| `mesh_generation.ipynb` | Building triangular fault meshes from traces, polygons, points, and slab2.0 grids |

## Testing

Example notebooks are heavy and data-dependent, so they are **run manually**,
not executed in CI (the Gorkha mesh has ~2800 patches, and each Green's-matrix
assembly takes tens of seconds). A lightweight smoke test in
`tests/test_examples.py` loads the bundled Gorkha fault and datasets through the
public API to catch format or API drift without running the full inversion.
