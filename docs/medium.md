# `geodef.medium` — Elastic half-space parameters

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

Every dislocation engine in GeoDef assumes a homogeneous isotropic elastic
half-space. `ElasticMedium` is the single declared home for that medium's
parameters. Attach one to a `Fault` and the same values flow through all of
its computations: Poisson's ratio reaches the Okada and triangular
dislocation kernels, and the shear modulus sets the default for
`fault.moment`, `fault.magnitude`, and `fault.stress_kernel`.

```python
import geodef

medium = geodef.ElasticMedium(shear_modulus=35e9, poisson_ratio=0.27)
fault = geodef.Fault.planar(lat=0.0, lon=100.0, depth=20e3,
                            strike=90.0, dip=30.0,
                            length=40e3, width=20e3,
                            n_length=4, n_width=2,
                            medium=medium)

fault.medium.mu            # 35e9 (alias for shear_modulus, Pa)
fault.medium.nu            # 0.27 (alias for poisson_ratio)
fault.medium.lame_lambda   # first Lame parameter, Pa
fault.medium.young_modulus # Young's modulus, Pa
```

If you never pass a medium you get `DEFAULT_MEDIUM`, a 30 GPa Poisson solid
(`nu = 0.25`) — the values GeoDef has always used.

## `ElasticMedium(shear_modulus=30e9, poisson_ratio=0.25)`

Immutable (frozen dataclass, hashable, comparable by value). Validation
happens at construction: `shear_modulus` must be positive and finite (Pa);
`poisson_ratio` must lie in `[0, 0.5)` — crustal rocks are typically
0.1–0.35.

Derived properties: `mu` and `nu` are short aliases; `lame_lambda` is
`2 mu nu / (1 - 2 nu)`; `young_modulus` is `2 mu (1 + nu)`.

## Where the medium is used

| Consumer | Parameter used |
|---|---|
| `fault.greens_matrix`, `fault.displacement`, `greens.greens` | `poisson_ratio` |
| `fault.stress_kernel` | both (`mu` scaling, `nu` in the strain kernel) |
| `fault.moment`, `fault.magnitude` | `shear_modulus` |

The explicit `mu=` arguments on `moment`/`magnitude`/`stress_kernel` remain
as per-call overrides; when omitted, the fault's medium supplies the value.

`fault.with_medium(new_medium)` returns a copy of a fault with different
elastic parameters (the immutable geometry arrays are shared). Fault files
store geometry only, so `Fault.load(..., medium=...)` applies the medium at
load time.

The low-level kernels (`okada85`, `okada92`, `tri`, and the functions in
`geodef.greens` / `geodef.gradients` / `geodef.bayes`) keep their plain
`nu` floats — the medium object is a `Fault`-level convenience, converted at
the interface.

## Caching

Green's-matrix and stress-kernel cache keys include the medium's parameters,
so faults that differ only in `poisson_ratio` or `shear_modulus` never share
cache entries (see [`cache.md`](cache.md)).
