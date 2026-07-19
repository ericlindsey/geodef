# GeoDef documentation

Start with the [five-minute quickstart](quickstart.md), then use the
[workflow map and decision guides](workflow.md) to choose the appropriate
level of API and scientific assumptions. The [glossary](glossary.md) is the
notation authority; [conventions](conventions.md) is the authority for units,
axes, signs, angles, and ordering.

## Learn by workflow

- [Tutorial course](../tutorials/README.md): fifteen equation-first chapters
  with separately executed worked solutions.
- [Worked examples](../examples/README.md): real or project-style workflows
  with provenance, assumptions, validation, and interpretation.
- [Usability baselines](usability.md): golden beginner workflows and tracked
  transformation/runtime metrics.

## Domain workflow

| Subject | Reference |
|---|---|
| Fault geometry and forward modeling | [fault](fault.md) |
| Slip packing and direction bases | [slip](slip.md) |
| GNSS, InSAR, and vertical datasets | [data](data.md) |
| Elastic material parameters | [medium](medium.md) |
| Linear inversion and assessment | [invert](invert.md) |
| Visualization and fit diagnostics | [plot](plot.md) |
| Input and physical validation | [validation](validation.md) |

## Matrices, geometry, and geodesy

| Subject | Reference |
|---|---|
| Green's matrices and regularization operators | [greens](greens.md) |
| Disk cache trust and invalidation | [cache](cache.md) |
| Local/geographic transformations | [transforms](transforms.md) |
| Triangular mesh generation | [mesh](mesh.md) |
| Euler poles and block velocities | [euler](euler.md) |
| Geographic map plotting | [geomap](geomap.md) |

## Physics kernels and advanced inference

| Subject | Reference | Requires |
|---|---|---|
| Okada dispatch and direct kernels | [okada](okada.md) | base |
| Array backend and precision | [backend](backend.md) | JAX optional |
| Differentiable geometry models | [gradients](gradients.md) | `geodef[jax]` |
| Bayesian slip and geometry posteriors | [bayes](bayes.md) | `geodef[bayes]` |

These flat Markdown pages are deliberately stable content homes. Repository
text search covers every page today; a generated site can add navigation and a
search index later without moving or duplicating content.
