# Beginner workflow baselines

This page defines the three golden beginner workflows and records observable
usability metrics. It is a review aid, not a substitute for automated tests or
human observation sessions.

## Golden workflows

1. **First forward model:** declare a `Fault`, named strike/dip slip arrays, and
   observation locations; call `fault.displacement`; make a labeled map/vector
   plot. Tutorials 00–01 execute this path.
2. **First inversion:** construct a named `GNSS` dataset with uncertainty; call
   `geodef.solve`; read `result.dip_slip` and `result.reduced_chi2`; call
   `plot.prediction(result)`. The [quickstart](quickstart.md) and tutorial 03
   execute this path.
3. **Joint GNSS + InSAR:** construct named datasets, pass both to `solve`, and
   use `invert.prediction` / `diagnostics` or result-aware plot functions for
   each partition. Tutorial 05 executes this path without manual row slices.

The complete quickstart was executed on the phase branch. Every canonical
chapter and every solution notebook was also executed from its own directory
with `MPLBACKEND=Agg` and the repository `src/` path.

## Metrics

Count a **manual transformation** when beginner-facing code concatenates slip,
slices a blocked result, slices stacked predictions, or reshapes patch order
without a named helper. Count an **ambiguous unit-bearing argument** when a
position-dependent value carries length, angle, or coordinate meaning without
its name establishing the unit/convention.

| Workflow | Required imports | Manual transforms in routine interpretation | Ambiguous unit-bearing positional arguments | Time to labeled diagnostic plot |
|---|---:|---:|---:|---:|
| Five-minute quickstart | 3 (`numpy`, `matplotlib`, `geodef`) | 0 | 0 | 1.8 s |
| First inversion chapter | 3 | 0 outside the explicit linear-algebra demonstration | 0 | 1.4 s |
| Joint GNSS + InSAR chapter | 3 | 0 outside the explicit stacking demonstration | 0 | 1.4 s |

Times are one local Apple-silicon execution in July 2026 and include kernel
startup; they are regression signals, not performance guarantees. The complete
base chapter set executed in well under the five-minute CI target. The JAX
geometry chapter took about 16 seconds including compilation; the compact
Bayesian chapter took about 4 seconds with the installed optional stack.

## Review checklist

- A learner reaches a labeled result before encountering blocked-vector layout.
- Ordering appears only in the chapters that teach the linear system.
- Every dataset has a stable name in joint diagnostics.
- Units and coordinate order are carried by keyword names.
- Error messages identify the invalid field and expected physical range.
- Each warning states what interpretation it changes and a remedy when one is
  available.

Periodic novice-geophysicist observation sessions remain a human project task.
Repeated confusion from those sessions should become an API, validation,
documentation, or golden-workflow regression test—not just another paragraph.
