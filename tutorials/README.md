# GeoDef Tutorials

These notebooks are the scaffolded introductory path for learning GeoDef with
synthetic data. They are executed by `tests/test_tutorials.py` as part of the
normal pytest suite.

| Notebook | What it covers |
|----------|----------------|
| `01_forward_model.ipynb` | Fault creation, Green's matrix assembly, and forward prediction |
| `02_caching.ipynb` | Hash-based caching for Green's matrices and stress kernels |
| `03_plotting.ipynb` | Slip, vector, InSAR, fit, 3-D, map, resolution, and uncertainty plots |
| `04_mesh_generation.ipynb` | Triangular mesh generation from traces, polygons, points, and slab2.0 grids |

Run the tutorial execution checks with:

```bash
uv run pytest tests/test_tutorials.py -q
```
