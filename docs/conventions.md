# GeoDef conventions

The single reference for coordinate axes, angles, units, array ordering, and
the regularization and misfit conventions. Every public geometry and data API
follows this page; when a published source uses a different convention, the
mapping is given here.

## Coordinate frames

- **Geographic:** longitude and latitude in degrees (WGS84), depth in meters
  **positive down**. When a function or array carries both horizontal
  coordinates, the documented order for named/new APIs is **`lon, lat`**
  (x, y order, as in GMT and GIS). Calls with several positional coordinate
  arrays are keyword-only precisely so the order cannot be confused.
  (`Fault.centers_geo` and `Mesh.centers_geo` follow this `[lon, lat, depth]`
  order; `Fault.centers_local` gives the same centroids in ENU meters.)
- **Local Cartesian:** East, North, Up (ENU) in meters, right-handed, tied
  to a `LocalFrame` that records origin latitude, longitude, altitude, and
  projection. The current projection identifier is `"wgs84-enu"` (WGS84
  geographic → ECEF → tangent ENU); it is explicit so later projections do
  not make saved or combined local arrays ambiguous. Anything named `*_enu`
  or `*_local` uses its object's `.frame`; `z`/`up` is negative below the
  surface, while `depth` is positive down. Use `LocalFrame.to_enu`,
  `to_geographic`, and explicit `transform_enu`/`geometry.to_frame` methods.
- Kernel-native frames (Okada's fault-aligned x/y, DC3D internals,
  triangular dislocation coordinates) never appear in public signatures;
  they are converted at the adapter layer inside `geodef.greens`.

## Angles

All angles are degrees.

- **Strike:** clockwise from North, 0–360. The fault dips to the right of
  the strike direction.
- **Dip:** from horizontal, 0–90.
- **Rake:** direction of hanging-wall slip measured in the fault plane,
  counterclockwise from the strike direction: 0 = left-lateral
  strike-slip, 90 = reverse (thrust), -90 or 270 = normal, 180 =
  right-lateral. `components='rake'` fixes one rake for all patches and is
  physically meaningful only when strike is uniform.
- **Slip azimuth:** geographic azimuth of horizontal slip, clockwise from
  North. `components='azimuth'` converts to each patch's local rake as
  `slip_azimuth - strike_i`, so it remains meaningful on curved meshes.
- **Plate rake:** a large-scale kinematic direction expressed in each patch's
  local strike/dip plane. Plate coordinates are rake-parallel and
  rake-perpendicular; unlike raw triangle-local components, they can remain a
  smooth basis across a variable-orientation mesh.

## Units

- Lengths, displacements, and slip: **meters**. Velocities: meters per
  chosen time unit (be consistent; uncertainties share the data's unit).
- Stress and moduli: **Pa**; seismic moment: **N·m**.
- Any API using a non-SI unit says so in its name or its argument names
  (e.g. `mesh` functions with `_km` parameters).
- Elastic parameters live on `geodef.ElasticMedium` (see
  [`medium.md`](medium.md)); the default is a 30 GPa Poisson solid.

## Array ordering

- **Slip vectors** are blocked: for `N` patches and both components, the
  first `N` entries are strike-slip, the last `N` dip-slip
  (`m[:N]`, `m[N:]`). Plate coordinates are likewise blocked
  `[rake_parallel | rake_perpendicular]`. Single-component bases (`'strike'`,
  `'dip'`, `'rake'`, `'azimuth'`) have length `N`. Use `geodef.slip` conversion
  functions or the named arrays on `InversionResult` to recover physical
  strike/dip components.
- **Green's matrix rows** follow each dataset's observation vector:
  GNSS with 3 components interleaves `[E, N, U]` per station (`[E, N]` for
  2-component data); InSAR contributes one LOS row per pixel; `Vertical`
  one row per station. Strain kernels order rows as documented by
  `Fault.greens_matrix`.
- **Multiple datasets** stack rows dataset-by-dataset in the order given;
  `geodef.greens.stack_obs` / `stack_weights` build the matching stacked
  observation vector and weight matrix.
- **Patch order** for structured rectangular grids varies along strike
  fastest: patch `k = i_strike + n_length * j_dip`; use
  `Fault.patch_index(strike_idx, dip_idx)` instead of hand-computing this.
  `Fault.reshape_patches` converts patch-first arrays to
  `[dip_index, strike_index, ...]`; `Fault.flatten_patches` reverses it.

## Regularization

GeoDef uses **one** convention everywhere (direct solves, `LinearSystem`,
L-curve, ABIC, cross-validation, geometry search, and fixed-lambda Bayesian
modes):

```
Phi(m) = (d - G m)^T W (d - G m)  +  lambda * ||L (m - m_ref)||^2
```

- `lambda` is the value of `regularization_strength`. It multiplies the
  *squared* seminorm — never `lambda^2`.
- The equivalent augmented least-squares system appends rows
  `sqrt(lambda) * L` (and data `sqrt(lambda) * L m_ref`).
- The normal equations read `(G^T W G + lambda L^T L) m = G^T W d (+ ...)`.
- The linear-Gaussian posterior covariance is
  `C_m = (G^T W G + lambda L^T L)^{-1}`.

Mapping from published sources:

| Source's convention | GeoDef equivalent |
|---|---|
| `alpha * \|\|L m\|\|^2` (e.g. Tikhonov texts) | `regularization_strength = alpha` |
| `alpha^2` or `lambda^2 * \|\|L m\|\|^2` (e.g. Aster et al.; earlier GeoDef tutorial drafts) | `regularization_strength = alpha^2` (their `alpha` is our `sqrt(lambda)`) |
| `(1/beta^2)` precision weighting (Bayesian, e.g. Fukuda & Johnson 2008) | `regularization_strength = sigma_d^2 / sigma_m^2` at fixed hyperparameters |

## Misfit statistics

- `chi2` always means the **unreduced** weighted sum of squared residuals
  `r^T W r`.
- `reduced_chi2` means `chi2 / dof`. For a whole inversion the degrees of
  freedom are `M - n_params`; per-dataset diagnostics use the effective
  DOF `n_obs - leverage` from the hat matrix.
- `rms` is unweighted, `wrms` weighted; both are root-mean-square residuals
  in the data's units.

## Depth sign quick reference

| Quantity | Sign |
|---|---|
| `depth`, `obs_depth`, fault files | positive down (m) |
| ENU `up` / `z` coordinates, `Mesh.vertices_enu` | positive up (m) |
| `Fault.planar(depth=...)` | centroid depth, positive down |

When converting: `depth = -up`.
