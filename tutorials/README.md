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

| Notebook | What it covers |
|----------|----------------|
| `01_forward_model.ipynb` | Fault creation, the linear forward model `d = G m`, predicting displacements (`G @ m` vs. `fault.displacement()`), moment magnitude |
| `02_discretization_and_g_matrix.ipynb` | Discretization, `G` as a design matrix, building `G` column by column vs. `greens_matrix()`, blocked-column / interleaved-row layout |

The full planned sequence (03 unregularized inversion, 04 regularization, 05
choosing regularization strength, 06 multiple datasets, 07 correlated noise, 08
bounds/constraints, 09 uncertainty/assessment, 10 nonlinear geometry) is tracked
in `PLAN.md` and added here as each notebook is written.

## Previous-generation notebooks (`old_*`)

`old_01_forward_model`, `old_02_caching`, `old_03_plotting`, and
`old_04_mesh_generation` are the earlier feature-oriented notebooks, kept as
reference while the progressive sequence above is built out. They are **not**
part of the executed test suite and may be removed once their material has been
folded into the new sequence or into dedicated reference notebooks.

## Running the execution checks

The live tutorial sequence is executed end-to-end by `tests/test_tutorials.py`
as part of the normal pytest suite:

```bash
uv run pytest tests/test_tutorials.py -q
```
