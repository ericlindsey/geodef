# `geodef.euler`

Euler pole estimation and rigid-block velocity prediction. Rigid-plate motion
on a sphere is a rotation about an Euler pole; this module fits that pole to
horizontal GNSS velocities and predicts the velocity field a pole produces.

**Conventions:** geodetic latitude/longitude in degrees, rotation rate in
degrees per million years (deg/Myr), horizontal velocities (East, North) in
mm/yr. Rotations are computed on a sphere (geodetic latitudes are converted
internally with `transforms.geod2spher`).

## Physical picture

For a rigid plate, velocity at Earth-fixed position `r` is
`v = omega x r`, where `omega` is the angular-velocity vector. The Euler pole
gives the latitude/longitude of that vector and its rotation rate. A good fit
supports rigid-block motion; coherent residuals after removing the pole can
indicate elastic strain accumulation, internal block deformation, reference
frame errors, or underestimated uncertainty. See
[Euler's rotation theorem](https://en.wikipedia.org/wiki/Euler%27s_rotation_theorem)
for background.

## Fitting a pole

### `best_fit_pole(lat, lon, ve, vn, sig_e, sig_n, rho=0.0) → (pole, cov_pole, chi2_reduced)`

Weighted least-squares fit of an Euler pole to horizontal GNSS velocities.

```python
from geodef import euler

pole, cov, chi2 = euler.best_fit_pole(lat, lon, ve, vn, sig_e, sig_n, rho=0.0)
lat_p, lon_p, rate = pole          # deg, deg, deg/Myr
```

`cov_pole` is the 3x3 covariance of the geodetic pole parameters and `chi2` is
the reduced chi-squared misfit (near 1 when the uncertainties are consistent
with the scatter). Requires at least two stations.

Pole latitude, longitude, and rate can be strongly correlated, especially for
a small network far from the pole. Inspect `cov_pole` and spatial residuals;
reduced chi-squared near one alone does not establish plate rigidity.

## Predicting and removing motion

### `pole_velocity(lat, lon, lat_p, lon_p, rate) → (ve, vn)`

Predict the horizontal velocity field (mm/yr) produced by a pole.

### `remove_pole(lat, lon, ve, vn, lat_p, lon_p, rate) → (ve_res, vn_res)`

Subtract a pole's rigid rotation from a velocity field — e.g. to view
velocities in a block-fixed frame and isolate near-fault deformation.

## Low-level helpers

| Function | Description |
|----------|-------------|
| `euler_vector(lat_p, lon_p, rate)` | Geodetic pole → scaled Cartesian rotation vector |
| `euler_location(omega)` | Cartesian rotation vector → `(lat, lon, rate)` |
| `euler_rot_matrix(lat, lon)` | `(2n, 3)` design matrix mapping a rotation vector to E/N velocities |
