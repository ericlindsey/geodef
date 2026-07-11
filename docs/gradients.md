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

## Physical and mathematical picture

A forward model predicts observations from model parameters. For a fault
geometry `theta` and slip `m`, write

```text
d(theta, m) = G(theta) m,
```

where `d` contains predicted displacements and `G(theta)` is the Green's
matrix. Slip is linear: doubling `m` doubles `d`. Geometry is nonlinear:
changing dip, depth, or a triangle vertex changes every relevant Green's
function.

A **Jacobian** collects local sensitivities. For geometry parameter `theta_j`,

```text
J[i, j] = partial d_i / partial theta_j.
```

Thus a small geometry perturbation has the first-order approximation
`delta d ~= J delta theta`. A Jacobian column answers a concrete question such
as “how does every predicted displacement change for a one-degree increase in
dip?” Its units follow the parameter: displacement per degree for dip,
displacement per meter for depth, and so on. See the
[Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)
and [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
for broader introductions.

Autodiff evaluates these derivatives by applying the chain rule to the forward
model. It avoids choosing a finite-difference step, but it does not make an
ill-conditioned inverse problem well constrained: data coverage, parameter
trade-offs, and regularization still control what can be inferred.

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

Coordinates use GeoDef's local convention: `e` is East, `n` is North, and `z`
is Up. Depth parameters are positive downward, while observation and triangle
vertex `z` coordinates are non-positive below the surface. Angles are degrees.

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
artefact-free configuration selection. The **rectangular** path is
reverse-mode safe: `jax.grad` of scalar misfits built on
`rect_displacement`/`rect_greens` is validated against `jacfwd` and
finite differences (this is what `geodef.bayes` differentiates), with
one caveat — dip gradients lose accuracy within ~0.01 degrees of exactly
vertical, from cancellation inherent in the published `1/cos(dip)`
formulas (both AD modes equally). If you build a scalar misfit and want
`jax.grad` over the **triangular** model, validate the gradients against
finite differences first.

All Jacobians are validated against central finite differences in
`tests/test_gradients.py`.

Forward mode is efficient here because there are only 7 rectangular geometry
inputs but often thousands of output displacement components. Reverse mode
(`jax.grad`) is usually preferable when a scalar objective depends on many
inputs. JAX's
[autodiff cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
explains this input/output-size trade-off.

---

## Full Green's matrix assembly `G(θ)`

```python
G = geodef.gradients.rect_greens(theta, e_obs, n_obs, n_length=8, n_width=4)
# (3*nobs, 2*N): rows [E, N, U] per point; columns [:N] strike-slip,
# [N:] dip-slip — same layout and patch ordering as Fault.planar +
# greens.displacement_greens, but flat-Cartesian and traceable in theta.

G = geodef.gradients.tri_greens(vertices, obs)
# vertices: (ntri, 3, 3) — traceable in every vertex coordinate.
# Triangles are vectorized with jax.vmap on the JAX backend.
```

`jax.jacfwd(rect_greens)(theta)` gives the sensitivity of every Green's
coefficient to the fault geometry — the core ingredient for
variable-projection geometry inversion, where slip is solved linearly
inside a nonlinear search over `theta`.

This separation is useful because a typical fault has a handful of geometry
parameters but tens to thousands of slip parameters. At each trial geometry,
GeoDef solves the large linear slip problem exactly and asks the nonlinear
optimizer to move only through the small geometry space.

Project onto InSAR line-of-sight (matches `InSAR.project`):

```python
G_los = geodef.gradients.los_project(G, look)   # look: (nobs, 3) [E,N,U]
```

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
