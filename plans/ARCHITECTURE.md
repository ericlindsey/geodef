# Architecture Improvements — Menu and Discussion

This is a menu of improvements to GeoDef's *internal structure*: work whose
purpose is to make the code safer to change, harder to misuse, and able to
support the features and capabilities in the companion menus
(`plans/FEATURES.md`, `plans/CAPABILITIES.md`). Almost nothing here changes
what a user's script computes; each item changes what it will cost to build
the next thing, or removes a way the package could silently produce a wrong
or confusing result.

Each item is written to stand alone: what it is in plain terms, why it
helps, what it costs, and when to do it. Items can be picked individually —
there is no required order beyond the stated dependencies. Detailed designs
carried over from the retired phase plans are kept with each item.

---

## The current architecture, briefly

It helps to have the shape of the package in mind. GeoDef is organized in
layers, where each layer may only depend on the layers below it (enforced
by `tests/test_layering.py`):

| Layer | Modules | Role |
|---|---|---|
| 6. Edges | `plot`, `geomap` | drawing; nothing depends on these |
| 5. Workflows | `invert`, `bayes` (`synthetic` will join) | user-facing problem solving |
| 4. Domain | `fault`, `data`, `mesh`, `euler` | the objects users hold |
| 3. Operators | `greens`, `geometry`, `slip`, `gradients` | matrices and transforms |
| 2. Kernels | `okada85`, `okada92`, `tri`, `okada` | published physics ports |
| 1. Foundation | `backend`, `validation`, `medium`, `transforms`, `cache` | shared plumbing |

A typical inversion flows top-down: `invert.solve` asks a `Fault` for its
Green's matrix, which is assembled by `greens` from the kernel ports, then
weighted, regularized, and solved in `invert._system`/`_solvers`. The
Priority 3 work already split the four largest modules into small private
files behind unchanged public names, added the layer test, and introduced a
private *engine registry* (`_engines.py`) describing what each deformation
engine can do.

The items below finish that program and add the load-bearing pieces the
feature menu needs.

---

## How to choose from this menu

| # | Item | What you get | Effort | When |
|---|---|---|---|---|
| A1 | Finish the engine registry | New engines become one registration, not a scavenger hunt | Small–medium | Anytime; before any new engine |
| A2 | Kernel adapter layer | Ports stay verifiable; JAX strain becomes possible | Medium | Before differentiable strain (F9) |
| A3 | Whitening + operator solve paths | Realistic InSAR sizes stop being impossible | Large | Before/with the noise & scale features (F3–F6) |
| A4 | Solver factorization & conditioning | Protection against silent digit loss; faster sweeps | Small–medium | After benchmarks; with large-problem work |
| A5 | Backend-state capture | A whole class of confusing JAX bugs prevented | Small | Soon |
| A6 | Sampler-independent results | Sampler libraries can change without breaking users | Small | Immediately before SMC (C1) |
| A7 | Boundary contracts | Cryptic deep-stack errors become named-argument errors | Small | Anytime |
| A8 | Top-level export trim (shipped) | The public API matches the documentation | Small (coordinated) | Before any public v0.2 tag |
| A9 | Schema-bump discipline | Old result files always load | Rule, not a task | Standing |

A practical reading: **A5 and A7 are small and protective — do them
soon in any order.** (A8, the export trim, has shipped.) The rest are
*enabling layers*: build each one
just-in-time with the first feature that needs it (noted per item), rather
than speculatively. That keeps every architecture commit paired with a
visible payoff and avoids building machinery that the eventual feature
turns out not to want.

---

## A1. Finish the engine registry and capability contracts

**What this is.** GeoDef has two deformation "engines": rectangular Okada
patches and triangular dislocations. Historically, code that needed to
behave differently per engine asked `if engine == "okada": ... elif
engine == "tri": ...` — and those branches were scattered across roughly
fifteen call sites in `fault`, `plot`, `invert`, and `geomap`. Adding a
third engine meant finding every one. The fix, already half done, is a
*registry*: one table (`src/geodef/_engines.py`) where each engine declares
its name, geometry type, the functions that compute its Green's matrices,
and its capabilities (surface/internal observation support, strain support,
autodiff support, how to draw its patch outlines):

```python
@dataclass(frozen=True)
class EngineSpec:
    name: str                      # "okada" | "tri"
    geometry: str                  # "rect" | "tri"
    displacement_greens: Callable
    strain_greens: Callable | None # None => capability absent
    surface: bool
    internal: bool
    autodiff: bool
    patch_outlines: Callable       # vertex arrays for plotting
```

Green's-matrix assembly already dispatches through this table, and
requesting an unsupported capability already raises a typed error naming
the engine, the missing capability, and the engines that have it.

**What remains.**

- The plotting and map code (`plot/_shared.py`, `plot/_fault_plots.py`,
  `geomap.py`) still holds per-engine branches that build patch outlines.
  Give the outline builders one shared home, move them into
  `EngineSpec.patch_outlines`, and delete the branches.
- The remaining cosmetic branches (`fault.areas`, `validate`, save-format
  defaults) then read `spec.geometry` instead of comparing strings.
- Write down the *shape* of the pluggable functions as private `Protocol`
  types (Python's way of saying "any function with this exact signature
  and meaning counts"): the two Green's-callable signatures, the
  regularization operator, whitening (defined here, implemented under A3),
  and solver callables. Private means they live in `_engines.py` /
  `_contracts.py` and carry no public stability promise yet.
- The registry itself **stays private** until at least two external
  engines exist. The Meade and point-source engines (capabilities menu C2,
  C3) are the acceptance test for making registration public.

**Why it helps.** Every new physics capability on the menu — Meade,
volcanic sources, layered half-space — arrives as one registration instead
of edits to fifteen files, and a user who asks a new engine for something
it cannot do gets a sentence, not a stack trace. Finishing the migration
now, while the branch inventory is fresh, is cheap; doing it alongside a
new engine port doubles the review burden of both.

**Cost and risk.** Small-to-medium; mechanical and behavior-preserving.
The existing tests plus the public-surface snapshots guard it. Risk: low.

---

## A2. Kernel adapter layer

**What this is.** `okada85.py`, `okada92.py`, and `tri.py` are deliberate
line-by-line ports of published scientific code. They are kept visibly
traceable to their sources — variable names, equation order, and all —
because that traceability *is* the verification argument. The price is
that they use internal coordinate conventions and singularity handling
that the rest of the package must tiptoe around. An *adapter* is a thin
wrapper that owns everything at the seam — coordinate conversion, input
shaping, singular-point masking — so that the port itself never needs to
be touched again, and callers never see kernel-native conventions.

**What remains.**

- Wrap each kernel entry point with a clearer adapter rather than
  cosmetically rewriting formulas (the open 3.2 roadmap item). The
  adapters become the single place where depth-sign and axis conversions
  happen.
- Audit the strain paths for JAX *traceability* in the adapter layer: no
  in-place array writes, no data-dependent Python branching, singularity
  handling via `backend.masked_eval` rather than boolean indexing. This is
  the prerequisite for differentiable strain kernels (features menu F9) —
  the fixes land in adapters, never in the ports.

**Why it helps.** Two things people rarely regret: the scientific
verifiability of the ports is preserved forever, and the JAX work stops
being blocked on "we'd have to modify the reference port." It also gives
new engines (C2, C3, C7) a consistent pattern to copy.

**Cost and risk.** Medium. The exact-equivalence golden tests (pytest
marker `exact`) make regressions loud. Risk: low-to-medium — the work is
subtle but well-guarded.

---

## A3. Whitening and operator-valued linear algebra

**What this is.** This is the largest and highest-leverage item on the
menu. Today the solver treats data noise as a dense covariance matrix: it
builds the full `(n, n)` array, inverts it, and multiplies through. For a
teaching problem with 100 observations that is instantly fine. For a
realistic InSAR scene with 50,000 points, the covariance matrix alone is
20 GB and inverting it is hopeless — the current path is a teaching tool
being asked to do production work.

The standard fix has two parts:

1. **Whitening instead of inversion.** "Whitening" means rescaling the
   data so its noise becomes unit-variance and uncorrelated — after which
   ordinary least squares is exactly correct. Crucially, you can *apply*
   a whitening to a vector without ever forming the inverse covariance
   matrix, and for structured noise (per-point sigmas, block-diagonal,
   sparse, or "smooth atmosphere + independent noise") applying it is
   cheap even at 50,000 points.
2. **Operators instead of stored matrices.** A `LinearOperator` is a rule
   for computing `G @ x` without storing `G`. Iterative solvers only ever
   need that rule, so the Green's matrix and the regularizer can stay
   implicit for very large problems.

**The design** (carried from the retired Phase 4 plan):

- New module `geodef.noise`, functions only, plus the one new public
  record this work adds:

  ```python
  @dataclass(frozen=True)
  class Whitening:
      apply: Callable[[np.ndarray], np.ndarray]  # x -> C^{-1/2} x
      logdet: float                              # log det C_d (ABIC needs it)
      n: int
  ```

  Constructors: `noise.diagonal(sigma)`, `noise.dense(cov)`
  (Cholesky-backed, SPD-validated), `noise.block_diagonal(parts)`,
  `noise.sparse(cov)` (`splu`; CHOLMOD used if importable, never
  required), `noise.low_rank(U, diag)` (`C = D + U Uᵀ` via Woodbury —
  the natural form for InSAR atmosphere).
- Datasets keep accepting `covariance=` exactly as today and additionally
  accept a `Whitening`; a new `DataSet.whitening` property returns the
  operator form either way.
- `LinearSystem` solves via whitened rows (`Gw = C^{-1/2}G`,
  `GᵀWG = GwᵀGw`) whenever any dataset carries an operator whitening. The
  dense-`W` code path is kept verbatim for dense input, so the teaching
  path stays bit-stable. `stack_weights` survives untouched as the
  educational helper.
- `LinearSystem` additionally accepts `LinearOperator`-compatible objects
  (anything with `shape`/`matvec`/`rmatvec`) for `G` and `L`; operator
  input simply requires an iterative solve method (A4/F6).
- **Parity tests are the contract**: every whitening form and every
  operator route must reproduce the dense route on teaching-sized
  problems to a stated tolerance (tight but not bit-identical —
  factorization order differs; rtol ≲ 1e-10 for well-conditioned float64).

**Why it helps.** This is the one structural change that lets the *same*
code teach a course on Tuesday and process a real InSAR scene on
Wednesday. Essentially the entire "scale to real data" half of the
features menu (F3 noise tools, F4 nuisance, F6 large problems) rides on
it, and the Bayesian machinery inherits the `logdet` it needs for honest
evidence terms.

**Cost and risk.** Large — the biggest single item here — but it slices
well: constructors first, dataset integration second, solver integration
third, each slice with its parity tests. Risk: medium; entirely managed by
the parity-test contract and by never touching the dense teaching path.

---

## A4. Solver factorization and conditioning defaults

**What this is.** The solver currently forms the *normal equations*
(`GᵀWG m = GᵀW d`). This is the textbook method and is fast, but it
squares the problem's condition number — a measure of how much input
noise gets amplified — which can silently cost you half your significant
digits on ill-conditioned problems. The alternatives (QR, SVD) are slower
but numerically gentler. Which one should be the default is an empirical
question, and the instruments to answer it now exist:
`LinearSystem.condition_report()` (shipped) and the benchmark harness in
`benchmarks/` (shipped).

**What remains.**

- Benchmark QR/SVD/Cholesky against the normal equations on the
  teaching-scale and realistic-scale seed problems; record accuracy and
  cost with the harness's full qualifiers.
- Decide the default from that evidence, and add a conditioning-triggered
  fallback (or at least a loud warning with the remedy) when
  `cond(G)²` approaches `1/eps` for the active precision — the threshold
  chosen from measured conditioning, not a guessed constant.
- Add the reusable *generalized eigendecomposition* of `(GᵀWG, LᵀL)`:
  one O(n³) factorization that makes every point of an L-curve/ABIC/CV
  sweep O(n²) afterward. This is the NumPy analogue of the existing
  batched JAX ABIC sweep and shares its tests (features menu F7 exposes
  the payoff).

**Why it helps.** Digit loss from squared conditioning is the classic
silent numerical failure — nothing crashes, the answer is just quietly
worse. This item converts that from an invisible risk into either a
measured non-issue or a handled case. The sweep factorization is a large,
free-feeling speedup for the hyperparameter selection everyone runs.

**Cost and risk.** Small-to-medium, explicitly evidence-first: no default
changes until the benchmarks justify them (a settled decision from the
Phase 3 review). Risk: low.

---

## A5. Backend-state capture (JAX hygiene)

**What this is.** `geodef.set_backend("jax")` flips a global switch. Any
object prepared *before* the flip — a `LinearSystem`, a Bayesian
posterior — silently holds arrays and compiled functions from the old
backend and precision. Using it after the switch can mix float32 with
float64 or trigger surprise recompilation, producing results that are
subtly wrong or mysteriously slow with no error anywhere.

**The plan.** Prepared objects record `(backend, precision)` at
construction; using one after a global switch raises an informative error
("this system was prepared under numpy/float64; rebuild it or restore the
backend"). Compiled kernels close over concrete dtypes at trace time
instead of reading global config mid-trace. `set_backend(...)` remains the
one simple entry point — this removes *hidden* dependence on the global,
not the convenience. Alongside it, a lightweight `backend.compile_report()`
(wrapper-based trace counting) plus a compilation-cache section in
`docs/backend.md`, so users can tell first-call compile cost from
steady-state cost and diagnose shape-churn.

**Why it helps.** This prevents a class of bug that is disproportionately
expensive for exactly the users the JAX surface targets — and the fix is
small. One open question from the Phase 5 draft: whether wrapper-based
trace counting is worth it versus just documenting `jax.log_compiles`;
prototype both on the geometry-search example and keep the simpler one.

**Cost and risk.** Small. Do it soon; it de-risks every later JAX item.

---

## A6. Sampler-independent result contract

**What this is.** Bayesian sampling currently means NUTS via the BlackJAX
library, and `PosteriorResult` reflects that heritage. Before a second
sampler arrives (SMC, capabilities menu C1), define the neutral contract
every sampler must fill: `draws`, `log_prob`, per-parameter `rhat`/`ess`
where meaningful, `seed`, `sampler` name, and a sampler-specific `extras`
mapping (NUTS: divergences, acceptance; SMC: weights, temperatures,
log-evidence). BlackJAX API details stay inside `bayes._sampling`;
`PosteriorResult` is extended compatibly (new optional fields, nothing
removed); `sample()`'s signature and behavior are unchanged.

**Why it helps.** BlackJAX changes its API more often than GeoDef should
change its user interface. The contract makes a BlackJAX upgrade — or a
whole additional sampler — a contained change instead of a breaking one.

**Cost and risk.** Small, but it must land *before* C1, not after: the
retrofit is what gets expensive. Risk: low.

---

## A7. Boundary contracts at module seams

**What this is.** The Priority 3 module splits created clean internal
seams (`invert._system` inputs, `invert._solvers` inputs). Add
lightweight shape/dtype/finiteness checks at those seams, reusing the
existing `validation` helpers, with messages that name the argument and
the expected shape/units. Trace-only kernel assertions stay private, per
the 0.2 policy.

**Why it helps.** A wrong-shaped array today can travel several layers
before NumPy notices, producing an error about broadcasting deep in a
private module. With seam checks, the error names the argument the user
actually passed. This is cheap insurance whose payoff shows up as support
questions that never get asked.

**Cost and risk.** Small; incremental; can be done seam-by-seam anytime.

---

## A8. Finish the top-level export trim (shipped)

**What this is.** The 1.6 API work trimmed `geodef.__all__` to the
beginner vocabulary but deliberately left the expert names (`lcurve`,
`abic_curve`, `model_covariance`, `LinearSystem`, `stack_obs`, and
friends) importable at top level as redundant aliases, so the notebooks
could be migrated once rather than twice. This has now landed: the
notebooks, examples, and `docs/*.md` use the module paths
(`geodef.invert.lcurve`, `geodef.greens.stack_obs`, ...), the top-level
aliases are deleted from `geodef/__init__.py`, and the change went in as
**one atomic commit** so the tutorial suite stayed green.
`tests/test_public_api.py` now asserts those names are **no longer
reachable** at the top level, and the "Transitional top-level aliases"
section has been removed from `docs/api_stability.md`.

**Why it helped.** The real public surface now matches the documented
one. This had to land **before any public v0.2 tag**, because a release
freezes the top level into a compatibility promise, and every day of
delay added new notebook cells and user habits to migrate.

**Cost and risk.** Small but coordinated — the one item here that
touched many files at once. Risk was low (mechanical, fully
test-guarded).

---

## A9. Schema-bump discipline (standing rule)

**What this is.** Result files (`InversionResult.save`) carry a schema
version with migration on load (the `_migrate_v2_regularization_keys`
precedent). Several menu items add result fields: noise scales and
convergence reports (F3/F6), nuisance parameters (F4), coupling views
(F2), and possibly persisted posterior results (C1). The rule: **batch
them into one version bump per release**, landed with the first item that
needs it and covering the rest, with a tested migration — not one bump
per feature.

**Why it helps.** Users' saved results keep loading forever, and the
migration code stays reviewable instead of becoming an archaeology of
tiny bumps. This is a coordination rule to hold while picking from the
menus, not a work item.

---

## Settled decisions carried forward

Decisions from the retired phase plans that remain binding:

1. Module splits use packages with `__init__.py` re-exports and private
   `_`-modules; the public path is unchanged (shipped, Priority 3).
2. `hypothesis` is a dev/test-only dependency; tests skip without it.
3. The engine registry is private until two external engines exist; no
   plugin framework before the callable contract is proven (A1).
4. No default-solver change without benchmark evidence (A4).
5. Reference-port interior names are private tier even where spelled
   without underscores — renaming would break source traceability (A2).
6. `Whitening` is the single new public record of the noise work; noise
   models are constructor functions, not a class hierarchy (A3).
7. Dense, transparent paths remain the defaults at teaching scale and are
   never deprecated by scale work (A3/A4).
