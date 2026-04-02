# CLAUDE.md - Agent Onboarding Guide

## Project Overview

**GeoDef** is a Python library for forward and inverse modeling of fault slip in elastic half-spaces. It targets both coseismic (earthquake) and interseismic (locked fault / coupling) applications. The core library is complete; remaining work is tutorial notebooks and future extensions.

**Read `PYTHON.md` before editing any code.**

---

## Repository Layout

```
geodef/
├── CLAUDE.md              # This file
├── PLAN.md                # Development roadmap
├── PYTHON.md              # Mandatory coding standards
├── pyproject.toml         # Package config (hatchling, src layout)
├── src/geodef/            # Installable package
├── tests/                 # 669 tests across 15 files
├── examples/              # Worked example notebooks (01–04)
├── docs/                  # Per-module API reference
├── geometry/              # Original Green's function sources (Matlab/Fortran/Python)
└── related/               # Reference code: shakeout_v2/, stress-shadows/, and docs
```

---

## Package Modules (`src/geodef/`)

| Module | What it provides |
|--------|-----------------|
| `okada85` | Surface displacements, tilts, strains (Okada 1985) |
| `okada92` | Internal deformation at depth (Okada 1992 / DC3D) |
| `tri` | Triangular dislocation displacements and strains (Nikkhoo & Walter 2015) |
| `okada` | Unified dispatcher: auto-selects okada85 (z=0) or okada92 (z<0) |
| `greens` | Green's matrix assembly, projection, stacking, Laplacian operators |
| `fault` | `Fault` class: factory methods, forward modeling, I/O, moment |
| `data` | `DataSet` base + `GNSS`, `InSAR`, `Vertical` data types |
| `invert` | Inversion: solvers, regularization, hyperparameter tuning, model assessment |
| `plot` | Visualization: slip, vectors, InSAR, fit, fault3d, map, resolution, uncertainty |
| `cache` | Hash-based disk caching for Green's matrices and stress kernels |
| `transforms` | Geodetic transforms: ECEF, ENU, geodetic, Vincenty, haversine |
| `mesh` | Triangular mesh generation: trace+dip, polygon, points, slab2.0 (optional deps) |

See `docs/` for per-module API reference with examples.

---

## Important Rules

- **Read `PYTHON.md` before editing any code.** Mandatory style guidelines, tooling requirements, and coding standards.
- Use red/green TDD. Write tests first, then write code to pass the tests.
- Use `uv` for package management. Use `pytest` for testing. Install with `uv pip install -e .`.
- All new functions must have type hints, docstrings, and tests.
- Use NumPy vectorization — avoid Python loops over observation points or fault patches.
- Coordinate convention: local Cartesian (x=East, y=North, z=Up). Green's functions may use internal conventions — always convert at the interface.
- Slip columns are blocked: `[:N]` strike-slip, `[N:]` dip-slip.

---

## Git Workflow

**Commit after every logical unit of work** — do not wait until a multi-step task is fully complete. Each commit should leave the test suite passing and represent a coherent, independently revertable change.

```bash
# Run tests before committing
uv run pytest

# Stage specific files (never `git add -A` blindly)
git add src/geodef/module.py tests/test_module.py

# Commit with a descriptive message
git commit -m "Short summary

Longer explanation of why the change was made.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

Commit granularity guidelines:
- New class or module → one commit
- Refactor of existing code → one commit (separate from feature additions)
- New tests → can be bundled with the code they test, or a follow-up commit
- Bug fix → one commit, referencing what was broken
- Do **not** bundle unrelated changes in a single commit

---

## Testing

```bash
uv run pytest
```

**769 tests passing** across 16 test files covering all modules. Reference data in `tests/reference_data/` — Matlab-generated `.npz` files for cross-validation of Green's function engines.
