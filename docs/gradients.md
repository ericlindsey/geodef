# `geodef.gradients` — Differentiable forward models

Exposes the rectangular (okada85) and triangular (Nikkhoo & Walter)
half-space forward models as JAX-differentiable functions, plus Jacobian
helpers. This is the foundation for gradient-based nonlinear geometry
inversion (the fast replacement for tutorial 10's grid search).

Requires the JAX backend:

```python
import geodef
geodef.backend.set_backend("jax")   # pip install geodef[jax]
```

---

## Differentiation variables

Each engine differentiates its **native** geometry parameterization:

- **Rectangles** — `theta = [e0, n0, depth, strike, dip, length, width]`:
  centroid position, orientation (degrees), and size.
- **Triangles** — the three **vertex coordinates** (shape `(3, 3)`).
  Vertices are the fundamental parameterization: they are well-defined
  for any mesh, including non-planar ones. To get gradients in terms of
  derived parameters (trace position, dip of a planar mesh, ...), write a
  `theta -> vertices` builder and compose — JAX chains through it
  automatically.

Both models are also differentiable in the slip vector, and `d` is linear
in slip, so the slip Jacobian is exactly the Green's-function basis.

---

## Forward models

```python
d = geodef.gradients.rect_displacement(theta, slip, e_obs, n_obs, nu=0.25)
# slip = [strike_slip, dip_slip, opening];  d has shape (nobs, 3) = [E, N, U]

d = geodef.gradients.tri_displacement(vertices, slip, obs, nu=0.25)
# obs has shape (nobs, 3) with z <= 0
```

These are plain traceable functions: use them inside your own
`jax.jit`/`jax.jacfwd`/optimization code, or compose them with a
parameter builder.

---

## Jacobians

```python
d, d_dtheta, d_dslip = geodef.gradients.rect_displacement_jacobian(
    theta, slip, e_obs, n_obs
)
# d: (nobs, 3);  d_dtheta: (nobs, 3, 7);  d_dslip: (nobs, 3, 3)

d, d_dvertices, d_dslip = geodef.gradients.tri_displacement_jacobian(
    vertices, slip, obs
)
# d_dvertices: (nobs, 3, 3, 3) — trailing axes index (vertex, coordinate)
```

Jacobians use **forward-mode** autodiff (`jax.jacfwd`), which suits the
many-observations / few-parameters shape of geometry inversion and avoids
the reverse-mode NaN-through-`where` pitfall in the triangular kernel's
artefact-free configuration selection. If you build a scalar misfit and
want `jax.grad` (reverse mode) over the triangular model, validate the
gradients against finite differences first.

All Jacobians are validated against central finite differences in
`tests/test_gradients.py`.

---

## Composing with a geometry builder

```python
import jax.numpy as jnp

def planar_vertices(theta):
    """Example theta -> vertices builder for a dipping triangle pair."""
    x0, y0, depth, dip_deg = theta
    dip = jnp.radians(dip_deg)
    return jnp.array([
        [x0, y0, -depth],
        [x0 + 8e3, y0, -depth],
        [x0, y0 + 5e3 * jnp.cos(dip), -depth - 5e3 * jnp.sin(dip)],
    ])

def predicted(theta, slip, obs):
    return geodef.gradients.tri_displacement(planar_vertices(theta), slip, obs)

# gradients with respect to the derived parameters, via the chain rule:
jac = jax.jacfwd(predicted)(theta, slip, obs)
```
