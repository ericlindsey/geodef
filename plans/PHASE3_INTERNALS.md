# Phase 3 Plan — Legible and Extensible Internals

This document is the **design plan for PLAN.md Priority 3**. It turns the
four roadmap items (3.1 layers, 3.2 module splits, 3.3 callable contracts,
3.4 numerical contracts) into commit-sized, behavior-preserving work with the
concrete file-level layout decided in advance. Revise this plan first if the
approach changes.

> **Status: drafted 2026-07, not yet started.** Written against the post-1.6
> v0.1 API while the Priority 2 course rewrite proceeds in parallel on its
> own branch.
>
> **Lifecycle.** Transient, like `tutorials/OUTLINE.md`: when Priority 3
> ships, settled decisions migrate to `PLAN.md` (as a short baseline entry)
> and `docs/api_stability.md`, and this file is retired to git history.

---

## 1. Hard Constraints

These bind every commit in this phase.

1. **Zero breaking changes.** Every public import path that works today
   works identically after this phase: `geodef.invert.solve`,
   `geodef.bayes.RectPosterior`, `geodef.plot.map_view`, and every other
   name keep their spelling. Splitting a module means converting it to a
   package whose `__init__.py` re-exports the full current surface.
2. **Numerical behavior is preserved exactly** where semantics are
   unchanged. Extractions are mechanical moves; the full test suite passes
   unchanged after every commit, and no test tolerance is loosened to make
   a refactor pass.
3. **The tutorials are unaffected.** No notebook, example, or `docs/*.md`
   file needs editing for this phase to land (documentation *additions* are
   fine). This keeps Phase 3 mergeable regardless of where the Priority 2
   rewrite stands.
4. **Small extraction commits.** One extracted submodule (or one contract,
   or one test family) per commit, each independently revertable, per the
   repository commit-granularity policy.
5. **No cosmetic rewrites of reference ports.** `okada85.py`, `okada92.py`,
   and `tri.py` stay visibly traceable to their published sources; they gain
   adapters and contracts, not style changes.

### Coordination with Priority 2 (parallel branch)

- Phase 3 touches `src/geodef/` and `tests/`; Priority 2 touches
  `tutorials/`, `docs/quickstart|glossary|workflow.md`, and one atomic
  commit editing `geodef/__init__.py` + notebooks (the 2.2 alias removal).
  The only shared file is `geodef/__init__.py`; Phase 3 does not change its
  public exports, so conflicts are mechanical at worst.
- The API stability map (§2) documents the tiers decided in 1.6. If it
  lands before the 2.2 alias-removal commit, the expert top-level aliases
  are listed as *"expert-public at their module path; top-level alias
  pending removal (PLAN.md 2.2)"* — the map describes the target state.

---

## 2. Item 3.1 — Layers and Public Boundaries

### Declared layering (dependency direction points down)

| Layer | Modules | May import |
|---|---|---|
| 6. Edges | `plot`, `geomap` | everything below |
| 5. Workflows | `invert`, `bayes`, `synthetic` (arrives with P2) | 1–4 |
| 4. Domain | `fault`, `data`, `mesh`, `euler` | 1–3 |
| 3. Operators | `greens`, `geometry`, `slip`, `gradients` | 1–2 |
| 2. Kernels | `okada85`, `okada92`, `tri`, `okada` | 1 |
| 1. Foundation | `backend`, `validation`, `medium`, `transforms`, `cache` | stdlib/NumPy/SciPy only |

Known wrinkles, resolved as follows:

- `fault` (layer 4) imports `greens` (layer 3) at runtime — correct
  direction. `greens` refers to `Fault` and `DataSet` only as type
  annotations; the rule is **lower layers may reference higher-layer types
  under `TYPE_CHECKING` only**, never at runtime.
- `cache` is foundation but calls `backend` for precision context
  (`cache.py:161`) — both are layer 1; intra-layer imports are allowed but
  must stay acyclic.
- No internal module may import through `geodef.__init__` at runtime.
  Current `import geodef` occurrences in `backend.py`, `cache.py`,
  `medium.py`, `gradients.py`, `bayes.py` are docstrings or
  `TYPE_CHECKING`; the new test locks this in.

### Work items (one commit each)

- [x] **`docs/api_stability.md`** — the published stability map. Three
  tiers: *beginner-public* (the `geodef.__all__` vocabulary), *expert-public*
  (stable at its module path: `invert.lcurve`, `greens.stack_obs`,
  `bayes.sample`, `LinearSystem`, …), *private* (`_`-prefixed or
  undocumented; no stability promise). One table of every public name with
  tier and home module; a short statement of the deprecation policy
  cross-referencing the changelog policy from 0.3. Link from `README.md`
  and `docs/` index.
- [x] **Tier-consistency test** — extend `tests/test_public_api.py` to
  parse the table in `docs/api_stability.md` and assert (a) every
  beginner-tier name is in `geodef.__all__` and vice versa, (b) every
  expert-tier name imports from its stated module path. The map can then
  never silently drift from the code.
- [x] **Import-graph test** — `tests/test_layering.py`: walk
  `src/geodef/*.py` with `ast`, extract runtime `geodef.*` imports
  (skipping `TYPE_CHECKING` blocks), and assert the edge set is a subset of
  the allowed matrix above. This is simultaneously the import-cycle test
  (the allowed matrix is acyclic by construction).
- [x] **Base-install / lazy-optional test** — in a subprocess: `import
  geodef` must succeed and must not import `jax`, `blackjax`, `cartopy`,
  `meshpy`, `pyproj`, or `matplotlib.pyplot`-at-import-time beyond what the
  current base install already does (measure first, then pin). Assert
  `sys.modules` after import. CI already tests install tiers (0.3); this
  adds the "importing geodef must not initialize JAX" guarantee explicitly.

---

## 3. Item 3.2 — Split Large Modules Behind Stable Re-exports

The four oversized modules, current sizes: `invert.py` 2965 lines,
`bayes.py` 2428, `fault.py` 1742, `plot.py` 1731. Each split converts the
module to a package (`geodef/invert/__init__.py` re-exporting everything) or
extracts private `_`-modules beside it. Public paths are unchanged either
way.

### Pre-split checks (do once, first)

- [x] Confirm nothing persists class module paths: result I/O is
  npz + JSON manifest (not pickle) — verify `save`/`load` and the cache
  payload format never store `__module__`/qualnames, so moving a class
  definition into a private submodule cannot break stored files. Add a
  round-trip test loading a result file saved before the split (a small
  fixture committed now, pre-split).
- [x] Snapshot the public surface: record `sorted(dir(geodef.<mod>))` for
  the four modules into a test fixture; after each split commit the
  snapshot must be unchanged (minus nothing, plus nothing public).

### 3.2a `invert.py` → `geodef/invert/` package — **done**

Target layout (line references are to today's file):

| New file | Contents (moved verbatim) |
|---|---|
| `_results.py` | `InversionResult` (58), `DatasetDiagnostics` (200), `LCurveResult` (821), `GeometrySearchResult` (893), `ABICCurveResult` (927) |
| `_io.py` | `save`/`load`/`save_table` and all manifest/migration helpers (223–819), incl. `_migrate_v2_regularization_keys` |
| `_system.py` | `LinearSystem` (1002–1802) with its methods (incl. `_abic_sweep_jax`), `_projection_matrix`, `_system_hash`, `_validate_args`, `_apply_weights` |
| `_regularization.py` | `_build_regularization_matrix`, `_build_reg_rhs` |
| `_solvers.py` | `_solve`, `_solve_constrained`, `_expand_bounds`, `_auto_select_method`, `_rank_positive_eigs`, `_compute_reduced_chi2`, `_physical_components` |
| `_selection.py` | `compute_abic`, `lcurve`, `abic_curve`, `_lcurve_corner` |
| `_geometry.py` | `geometry_search`, `_vp_residual`, `_vp_residual_and_jacobian`, `_vp_kernel`, `_fault_from_planar_vector` |
| `_assessment.py` | `prediction`, `residual`, `diagnostics`, `summary`, `model_covariance`, `model_resolution`, `model_uncertainty` |
| `__init__.py` | `solve` (thin: builds `LinearSystem`, calls `invert`) plus re-exports of the entire current public surface |

Internal dependency direction: `_results` ← everything; `_system` ←
`_results`, `_regularization`, `_solvers`; `_selection`/`_assessment`/
`_geometry` ← `_system`. One commit per extracted file, in the order
`_results`, `_io`, `_solvers`+`_regularization`, `_selection`,
`_assessment`, `_geometry`, leaving `_system` and the package conversion
last.

### 3.2b `bayes.py` → `geodef/bayes/` package — **done** (plus a
`_util.py` for the shared jax-guard/slip-transform/parse helpers)

| New file | Contents |
|---|---|
| `_collapsed.py` | `_CollapsedPosterior` (130), `RectPosterior` (545) |
| `_triwarp.py` | `TriWarp` (1137), `_parse_knot_prior`, `TriPosterior` (1428) |
| `_slip.py` | `_slip_transform` (87), `SlipPosterior` (1676), `_parse_positive` |
| `_sampling.py` | `PosteriorResult` (2206), `sample` (2305), `_require_jax` |
| `_diagnostics.py` | `_split_chains`, `split_rhat` (2140), `effective_sample_size` (2163) |

`_sampling` ← `_diagnostics`; posterior modules are siblings that share
only `_slip_transform`. This layout is also the landing zone for Phase 5
additions (`sample_smc` goes in `_sampling`; predictive checks in a new
`_checks.py`), so it should land before 5.2/5.3 begin.

### 3.2c `fault.py` — extract I/O, keep the class whole — **done**

`Fault` itself (geometry, factories, forward conveniences, properties) is
cohesive and stays in `fault.py`. Extract:

- [x] `_fault_io.py`: the format-specific loaders/savers
  (`_load_center`, `_load_topleft`, `_load_seg`, `_save_center`,
  `_save_seg`, `_save_tri_ned`, `to_gmt` body, `_seg_to_patches`).
  `Fault.load`/`Fault.save`/`Fault.to_gmt` become thin dispatchers. The
  skipped-when-absent `related/stress-shadows` load tests must still pass
  untouched where reference data exists.
- [x] Keep `moment_to_magnitude`/`magnitude_to_moment` re-exported from
  `fault` (public today).

### 3.2d `plot.py` → `geodef/plot/` package — **done**

| New file | Contents |
|---|---|
| `_shared.py` | axes helpers, `_stations_to_local_km`, the patch-vertex builders (31–520), `_get_slip_component`, `_plot_patch_scalar`, `_get_surface_trace`, `_add_scale_arrow_legend` |
| `_fault_plots.py` | `patches`, `slip`, `slip_interpolated`, `fault3d` |
| `_data_plots.py` | `vectors`, `insar`, `map_view` |
| `_fit_plots.py` | `fit`, `prediction`, `residual`, `diagnostics`, `summary` |
| `_assessment_plots.py` | `resolution`, `uncertainty` |

### 3.2e `data.py` — deduplicate I/O only

`data.py` (1703 lines) is below the split threshold once its shared I/O
helpers are already factored (they are: `_names_header`, `_read_metadata`,
…). Work here is limited to:

- [x] Audit the three dataset classes' `save`/`load` for residual
  copy-paste and pull any remaining duplication into the existing helper
  layer. **No inheritance deepening** — helpers stay module functions.

### Verification per split commit

Full routine suite (`uv run pytest`) plus the surface-snapshot fixture and
a smoke import of every module path the notebooks use. `ruff` and `mypy`
stay clean (update per-module ignores if any are file-scoped).

---

## 4. Item 3.3 — Callable Contracts Instead of String Dispatch

### Current state (inventory)

`engine == "okada"` / `"tri"` branches live at: `fault.py` 97/103/773/799/
975/1006/1256/1477, `plot.py` 130/191/446/486/726, `invert.py` 773,
`geomap.py` 104. `greens.py` has parallel rect/tri function pairs
(`displacement_greens`/`tri_displacement_greens`, likewise strain). Adding
an engine today means finding all fifteen branch sites.

### Design

- [ ] **Private engine registry** — new `src/geodef/_engines.py`
  (foundation-adjacent, layer 3):

  ```python
  @dataclass(frozen=True)
  class EngineSpec:
      name: str                      # "okada" | "tri"
      geometry: str                  # "rect" | "tri"
      displacement_greens: Callable  # (fault, stations, medium, ...) -> G_raw
      strain_greens: Callable | None # None => capability absent
      surface: bool                  # z = 0 observation support
      internal: bool                 # z < 0 observation support
      autodiff: bool                 # differentiable path exists (gradients)
      patch_outlines: Callable       # (fault) -> vertex arrays for plotting
  ```

  with `register(spec)` / `get(name)` and the two built-ins registered at
  import. `Fault.engine` remains the public string; user-facing behavior
  and error text for *valid* inputs are unchanged.
- [ ] **Migrate call sites** in three commits: (a) `greens` assembly picks
  kernels from the spec; (b) `fault` forward/moment/save paths; (c)
  `plot`/`geomap` use `patch_outlines` (this is what their engine checks
  actually select). `invert.py:773` follows (b).
- [ ] **Capability errors** — requesting an unsupported combination (e.g.
  strain from an engine with `strain_greens=None`, internal points from a
  surface-only engine) raises a typed error naming the engine, the missing
  capability, and the engines that have it.
- [ ] **Typed contracts** — `Protocol` types for the two Green's-callable
  signatures, the regularization operator (already accepts a matrix; the
  contract documents shape and symmetry expectations), whitening (defined
  here, implemented in Phase 4), and solver callables. These start
  **private** (`_engines.py` / `_contracts.py`); a public protocol is
  promoted only when an external engine wants it, per the roadmap's
  no-premature-plugin rule.
- [ ] **Registry stays private this phase.** Public `register_engine` is
  deferred until at least two external engines exist (Phase 6.2 provides
  the first real candidates — Meade tri, point sources — and is the
  acceptance test for the contract).

---

## 5. Item 3.4 — Numerical Contracts

- [ ] **Property-based tests** (new dev-only dependency: `hypothesis`,
  settled below): round trips `slip.pack`/`unpack` and
  `fault.reshape_patches`/`flatten_patches`; `slip.from_rake`/
  `from_azimuth` produce unit-magnitude bases; `transforms` geodetic ↔
  ECEF ↔ ENU round trips within stated tolerance; `greens.project`
  linearity (`project(a·x + b·y) == a·project(x) + b·project(y)`) and row
  count; `greens.matrix` linearity in slip against `fault.displacement`.
- [ ] **Boundary contracts** — lightweight shape/dtype/finite checks at the
  module seams that 3.2 exposes (`_system` inputs, `_solvers` inputs),
  reusing `validation` helpers; trace-only asserts stay private per 0.2.
- [ ] **Conditioning diagnostics** — `LinearSystem` gains a
  `condition_report()` (cond of the whitened `G`, cond of
  `H = GᵀWG + λLᵀL` at the chosen λ, rank estimate) surfaced through
  `invert.summary`; a warning when cond(G)² approaches 1/eps for the
  active precision. **No default-solver change in this phase**: the normal
  equations remain; the QR/SVD/Cholesky benchmark below produces the
  evidence, and any switch is decided with Phase 4.2's solver work.
- [ ] **Test taxonomy** — pytest markers `exact` (bit/ULP equivalence:
  golden okada92 outputs, refactor-equivalence), `physical`
  (tolerance-based cross-validation: Matlab reference data), `benchmark`
  (excluded from the default run). Document tolerance provenance where the
  markers are applied; no tolerance values change.
- [ ] **Benchmark harness** — `benchmarks/` (not shipped in the wheel): a
  small runner that records problem definition (patches, stations,
  datasets), compile vs steady-state time, peak memory (`tracemalloc`),
  backend, precision, NumPy/JAX versions, and hardware stamp to JSON.
  Seed problems: teaching scale (~80 patches / ~100 obs) and realistic
  scale (~2 000 patches / ~20 000 obs). Runs on demand, not in CI (a CI
  smoke invocation only checks the harness itself executes). This is the
  instrument Phase 4.2 uses to document the scale boundary.

---

## 6. Suggested Commit Sequence

1. Stability map + tier test (3.1) — no `src/` changes.
2. Layering + base-install tests (3.1) — locks the ground before moving it.
3. Pre-split checks and snapshots (3.2).
4. `invert` split, ~7 commits (3.2a).
5. `bayes` split, ~4 commits (3.2b) — unblocks Phase 5 file layout.
6. `fault` I/O extraction; `plot` split; `data` dedup (3.2c–e).
7. Engine registry + call-site migration, ~4 commits (3.3).
8. Property tests, conditioning report, markers, benchmark harness (3.4).

Steps 1–2 and 8 can interleave anywhere; 4–7 should not run concurrently
with each other on separate branches (merge conflicts in the same files).

---

## 7. Settled Decisions

1. **Splits use packages with `__init__.py` re-exports**, private
   `_`-prefixed submodules, and no deprecation machinery — the public path
   *is* the package path, unchanged.
2. **`hypothesis` is added as a dev/test dependency only.** It must not
   appear in any install extra that users see; tests using it are skipped
   if it is absent (so the base CI tier stays honest).
3. **The engine registry is private in this phase.** Public registration
   waits for a second external engine (roadmap rule), i.e. Phase 6.2.
4. **No solver change under 3.4.** Diagnostics and benchmarks only;
   changing the default factorization is a Phase 4.2 decision informed by
   this phase's measurements.
5. **`fault.py` is not converted to a package** — only its I/O moves out.
   The class-heavy remainder (~1 200 lines) is cohesive; a package there
   would be structure without benefit.
6. **The `plot` split (3.2d) proceeds despite the in-flight Priority 2
   branch having permission to edit `plot.py`** (user decision, 2026-07).
   If P2 lands `plot.py` edits after the split, the resolution is
   mechanical: re-apply those edits to the function's new home in
   `geodef/plot/_*.py` — expect and budget for that conflict rather than
   serializing the phases. (Resolves former open question 2.)
7. **The stability map lists reference-port interior names as private
   tier** even though they are spelled without underscores (`okada85`
   integrands, `tri` angular-dislocation helpers): renaming them would
   break traceability to the published sources. Only the kernel entry
   points are expert-public.

## 8. Open Questions

1. **Where does `synthetic` (built by Priority 2) sit in the layer table?**
   Proposed: layer 5 (workflows) since it composes `fault` + `data` +
   forward models. Resolve when the module lands on the P2 branch; the
   layering test's allowed matrix is updated in the same commit.
2. *(Resolved — settled decision 6.)*
3. **Condition-number warning threshold** — cond(G)² vs 1/eps with what
   safety factor, and warn-once vs per-solve? Decide from the benchmark
   problems' measured conditioning rather than picking a constant now.
