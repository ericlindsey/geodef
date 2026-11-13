# CLAUDE.md — working on GeoDef

GeoDef is a Python library for forward and inverse modeling of fault slip in
an elastic half-space, for earthquakes (coseismic slip) and locked faults
(interseismic coupling). It is used for **teaching**: a fifteen-chapter
tutorial course is built on it, and students read the source.

That sets the priority order for every change:

1. **Correct physics.** Pin behavior with physical tests (see Testing).
2. **One obvious way to do each thing**, with the same names everywhere.
3. **Code a student can follow** from the public call down to the kernel.
4. Research power (JAX, Bayesian sampling) stays optional and out of the way.

The API is pre-1.0 and has no compatibility promise. When a name or structure
is wrong, change it everywhere (code, tests, notebooks, docs) and record it in
`CHANGELOG.md`; do not add aliases or deprecation shims.

---

## The API on one screen

```python
import geodef

fault = geodef.Fault.planar(lon=, lat=, depth=, strike=, dip=,
                            length=, width=, n_length=, n_width=)
east, north, up = fault.displacement(lon=, lat=, strike_slip=, dip_slip=)

gnss  = geodef.GNSS(lon=, lat=, east=, north=, up=, sigma=)
insar = geodef.InSAR(lon=, lat=, los=, sigma=, look_e=, look_n=, look_u=)
gnss  = geodef.GNSS.synthetic(fault, lon=, lat=, dip_slip=, sigma=, rng=0)

result = geodef.solve(fault, [gnss, insar], components="dip",
                      regularization="laplacian", regularization_strength=1.0)
result.dip_slip, result.reduced_chi2, result.uncertainty(), result.resolution()

system = geodef.LinearSystem(fault, gnss, regularization="laplacian")
system.G, system.d, system.lcurve(), system.abic_curve(), system.solve(...)

geodef.plot.slip(fault, result.dip_slip)
```

Rules that keep it unified. Follow them when adding anything:

- **Coordinates are keyword-only `lon=`, `lat=`**, in that order, in degrees.
  Never add a positional `(lat, lon)` or `(lon, lat)` signature; swapped
  coordinates fail silently.
- **Slip is `strike_slip` / `dip_slip`**, one value per patch (scalars
  broadcast), on inputs and on results. The blocked vector
  `[strike-slip (N) | dip-slip (N)]` appears only where the linear algebra is
  the subject (`G`, `slip_vector`, `geodef.slip.pack`).
- **Datasets have one constructor each.** `name` is the dataset's name,
  `station_names` the per-station labels. Scalars broadcast over stations.
- **A result knows its problem.** Assessment is a method on the result
  (`result.covariance()`), never a free function that takes the fault and
  data again.
- **`solve` for one answer, `LinearSystem` for many.** Do not add a third path.
- Top-level names are the everyday vocabulary (`geodef.__all__`); everything
  else is reached through its module (`geodef.plot.slip`, `geodef.mesh.from_trace`).

## Conventions

- Local Cartesian is East, North, Up in meters. Depth is positive down.
  Angles are degrees. SI units throughout; anything else says so in its name.
- Strike is clockwise from north and **the fault dips to the right of strike**.
- Planar grids are stored row by row: **row 0 is the shallowest**, each later
  row is one step down dip; patches advance along strike within a row.
  `fault.reshape_patches` / `fault.patch_index` name this layout.
- GNSS observations are interleaved by station: `[e1, n1, u1, e2, ...]`.
- `docs/conventions.md` is the full reference; keep it authoritative.

---

## Layout

```
src/geodef/        the package
tests/             one file per module, plus executed notebooks and docs
tutorials/         chapters 00-14, solutions/, reference_plots.ipynb (plot gallery)
examples/          real-data and project examples (Gorkha, meshes, Bayesian geometry)
docs/              per-module reference, quickstart, conventions, glossary
plans/, PLAN.md    forward-looking roadmap
related/, geometry/  legacy and reference sources (read-only; not linted)
```

| Module | What it holds |
|--------|---------------|
| `fault` | `Fault`: geometry, forward model, moment, file I/O (`_fault_io`) |
| `data` | `DataSet`, `GNSS`, `InSAR`, `Vertical`, `spatial_covariance` |
| `invert` | `solve`, `LinearSystem`, `InversionResult`, `geometry_search` |
| `plot`, `geomap` | Matplotlib plots; optional Cartopy maps |
| `greens` | Green's matrix assembly, dataset projection, Laplacians |
| `okada85`, `okada92`, `tri`, `okada` | Published dislocation kernels (line-by-line ports) |
| `slip` | Blocked-vector packing and rake/azimuth/plate bases |
| `geometry`, `transforms` | `LocalFrame`; geodetic conversions |
| `medium`, `validation`, `cache` | Elastic constants; input checks; disk cache for `G` |
| `mesh`, `euler` | Triangular mesh generation; Euler poles |
| `backend`, `gradients`, `bayes` | Optional JAX backend, autodiff models, NUTS sampling |

How a solve flows: `geodef.solve` → `LinearSystem.__init__` (assemble `G`,
`d`, `W`, `L`) → `greens.matrix` → `Fault.greens_matrix` →
`greens.displacement_greens` → `okada85.displacement`.

---

## Working rules

- **Do the work directly.** No subagents unless the user asks for them.
- **Red/green TDD.** Write the failing test, then the code.
- **Prefer physical tests to formula tests.** A test that re-implements the
  code's formula passes when both are wrong: the reversed row depths in
  `Fault.planar` survived for that reason. Assert invariants instead:
  subdividing a fault does not change uniform-slip displacement, adjacent
  patches share an edge, a result reproduces the slip that made the data.
- **Keep the kernels verbatim.** `okada85.py`, `okada92.py`, and `tri.py`
  mirror their published sources and are excluded from formatting. Fix
  conventions at the call site, not inside them.
- **NumPy is the default.** Vectorize over stations and patches. JAX,
  blackjax, Cartopy, meshpy, and pyproj are optional extras; the base install
  must import and pass without them.
- **Notebooks are part of the API.** A rename is not done until
  `tutorials/`, `tutorials/solutions/`, `examples/`, and `docs/` use it.
- **Update `PLAN.md`** when a roadmap item ships or its design changes.
- **No `Co-Authored-By` trailers.** AI co-authorship is recorded once in
  `README.md`.

### Code style

- Type hints and Google-style docstrings (Args / Returns / Raises) on every
  public function, class, and method. State units and array shapes.
- Ruff for linting and formatting (88 columns); mypy must pass.
- Fail early with a `ValueError` that names the argument, what was received,
  and the expected unit or range (`geodef.validation` has the helpers).
- No emoji. No commented-out code. No comments that restate the code.
- Add a class only when it owns state or an invariant; otherwise a function.
- `orjson` for JSON. `logger.error`, not `print`, for errors.

### Commands

```bash
uv sync --dev --all-extras        # environment with every optional stack
uv run pytest tests/test_fault.py # the module you touched
uv run pytest                     # routine suite, about 9 minutes
uv run pytest -m slow             # adds solution notebooks and long sampling runs
uv run ruff check src tests && uv run ruff format src tests
uv run mypy
```

The routine suite executes every tutorial chapter, the plot gallery, the
quickstart in `docs/quickstart.md`, and the Gorkha example. Markers: `slow`,
`benchmark` (both excluded by default), `exact` (bit-level golden outputs; do
not loosen), `physical` (tolerance-based reference comparisons).

### Commits

Commit each logical unit with its tests passing: a bug fix, a rename, a new
capability. Keep refactors separate from behavior changes. Stage specific
paths, write a message that explains why, and push the working branch.

---

## Where complexity remains

Known debts, in rough order of what a student would trip over. Each is a
candidate for the next simplification pass (see `PLAN.md`):

- `ref_lat`/`ref_lon` and `frame=` are both accepted by `from_triangles`,
  `geometry_search`, `bayes`, and `mesh`; `LocalFrame(lat, lon)` is positional
  and latitude-first, against the rule above.
- `greens.py` has a per-patch NumPy loop and a separate batched JAX path for
  the same assembly, and its public functions take ten positional arguments
  in `(lat, lon)` order.
- `InversionResult` has more than thirty fields, most of them provenance.
- `invert/_io.py` carries a versioned, checksummed result-file schema with
  migrations.
- Rectangular patches use a per-patch tangent-plane projection; triangles use
  one shared `LocalFrame`.
