# CLAUDE.md - Agent Onboarding Guide

## Project Overview

**GeoDef** is a Python library for forward and inverse modeling of fault slip
in elastic half-spaces. It targets both coseismic (earthquake) and interseismic
(locked fault / coupling) applications. As of **v0.1** the runtime library, the
fifteen-chapter tutorial course, the per-module documentation, and the optional
JAX accelerator (differentiable forward models, gradient-based
`geometry_search`, and the collapsed Bayesian sampler `geodef.bayes`) are
complete; `ruff` and `mypy` pass cleanly and the suite runs warning-free.
Remaining forward-looking work is organized in `PLAN.md` plus three menu
documents: `plans/ARCHITECTURE.md`, `plans/FEATURES.md`, and
`plans/CAPABILITIES.md`.
Version 1.0 is reserved for completion of that roadmap and human testing.

**Read `PYTHON.md` before editing any code.**

---

## Repository Layout

```
geodef/
├── CLAUDE.md              # This file
├── PLAN.md                # Development roadmap (indexes the plans/ menus)
├── plans/                 # Menu documents: architecture, features, capabilities
├── PYTHON.md              # Mandatory coding standards
├── pyproject.toml         # Package config (hatchling, src layout)
├── src/geodef/            # Installable package
├── tests/                 # Test suite (one file per module plus integration)
├── tutorials/             # Fifteen-chapter teaching course executed by pytest
├── examples/              # Project and real-data examples
├── docs/                  # Per-module API reference
├── geometry/              # Original Green's function sources (Matlab/Fortran/Python)
└── related/               # Reference code: shakeout_v2/, stress-shadows/, and docs
```

---

## Package Modules (`src/geodef/`)

| Module | What it provides |
|--------|-----------------|
| `backend` | Array backend selection: NumPy (default) or JAX, precision control |
| `okada85` | Surface displacements, tilts, strains (Okada 1985) |
| `okada92` | Internal deformation at depth (Okada 1992 / DC3D) |
| `tri` | Triangular dislocation displacements and strains (Nikkhoo & Walter 2015) |
| `okada` | Unified dispatcher: auto-selects okada85 (z=0) or okada92 (z<0) |
| `greens` | Green's matrix assembly, projection, stacking, Laplacian operators |
| `gradients` | Differentiable forward models: Jacobians w.r.t. geometry and slip (JAX) |
| `fault` | `Fault` class: factory methods, forward modeling, I/O, moment |
| `slip` | Slip-vector packing and strike/dip, rake, azimuth, and plate-basis conversions |
| `medium` | `ElasticMedium`: shear modulus and Poisson's ratio, shared by Green's functions, stress kernels, and moment |
| `data` | `DataSet` base + `GNSS`, `InSAR`, `Vertical` data types |
| `invert` | Inversion: solvers, fixed-direction slip bases, regularization, hyperparameter tuning, model assessment, scalar/per-component/per-parameter bounds |
| `bayes` | Bayesian inference: collapsed rect/tri-mesh geometry posteriors (`RectPosterior`, `TriWarp`+`TriPosterior`), joint slip sampling with positivity (`SlipPosterior`), NUTS sampling (blackjax), slip credible intervals (JAX) |
| `plot` | Visualization: slip, interpolated slip, vectors, InSAR, fit, fault3d, map, resolution, uncertainty |
| `geomap` | Optional Cartopy geographic map plotting (basemap, fault/vector overlays) |
| `cache` | Hash-based disk caching for Green's matrices and stress kernels |
| `transforms` | Geodetic transforms: ECEF, ENU, geodetic, Vincenty, haversine |
| `validation` | Fail-early input checks and `.validate()` reports (`ValidationReport`) |
| `mesh` | Triangular mesh generation: trace+dip, polygon, points, slab2.0 (optional deps) |
| `euler` | Euler pole fitting and rigid-block velocity prediction |

See `docs/` for per-module API reference with examples.

---

## Important Rules

- **Read `PYTHON.md` before editing any code.** Mandatory style guidelines, tooling requirements, and coding standards.
- Work directly on coding tasks. Do not use subagents unless the user explicitly
  requests delegation or parallel agent work.
- Use red/green TDD. Write tests first, then write code to pass the tests.
- Use `uv` for package management. Use `pytest` for testing. Install with `uv pip install -e .`.
- All new functions must have type hints, docstrings, and tests.
- Use NumPy vectorization — avoid Python loops over observation points or fault patches.
- Coordinate convention: local Cartesian (x=East, y=North, z=Up). Green's functions may use internal conventions — always convert at the interface.
- Slip columns are blocked: `[:N]` strike-slip, `[N:]` dip-slip.
- When executing a step listed in `PLAN.md`, update `PLAN.md` in the same
  logical unit so the roadmap remains current.
- Keep the docs `.md` files up to date when making code changes. Minor docs
  fixes should land with the code; large docs rewrites should be added as their
  own `PLAN.md` step.
- Do not add `Co-Authored-By` trailers to commit messages. AI co-authorship is
  tracked once in `README.md`; update that model list, this file, and
  `AGENTS.md` if a new AI model materially contributes.

---

## Git Workflow

**Commit after every logical unit of work** — do not wait until a multi-step task is fully complete. Each commit should leave its relevant tests passing and represent a coherent, independently revertable change.

Commit messages should describe the change without AI co-author trailers.

```bash
# Run the tests relevant to this contained change before committing
uv run pytest tests/test_module.py

# Stage specific files (never `git add -A` blindly)
git add src/geodef/module.py tests/test_module.py

# Commit with a descriptive message
git commit -m "Short summary

Longer explanation of why the change was made."
```

Commit granularity guidelines:
- New class or module → one commit
- Refactor of existing code → one commit (separate from feature additions)
- New tests → can be bundled with the code they test, or a follow-up commit
- Bug fix → one commit, referencing what was broken
- Do **not** bundle unrelated changes in a single commit

---

## Testing

For a small, contained commit, run the directly relevant test module(s) or test
selection. Expand the selection when shared interfaces or cross-module behavior
are affected. Run the full routine suite when wrapping up a pull request or a
major change:

```bash
uv run pytest
```

The suite covers every module (do not hard-code collected-test counts
here; they drift). Reference
data in `tests/reference_data/` — Matlab-generated `.npz` files for
cross-validation of Green's function engines, plus golden okada92 outputs
captured from the pre-vectorization scalar port. A few `Fault.load` tests
need reference data under `related/stress-shadows/` and are skipped when it
is absent; the JAX/blackjax-gated backend, gradient, and Bayesian tests are
skipped when those optional dependencies are not installed.
