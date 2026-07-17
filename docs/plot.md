# `geodef.plot` — Visualization

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

All plot functions follow a consistent pattern:

- Accept optional `ax=None`; create a new figure if `None`
- Return the `matplotlib.axes.Axes` used
- Never call `plt.show()`
- Pass `**kwargs` through to the underlying matplotlib artist

## Plotting as a diagnostic

A good inversion figure should show observations, predictions, residuals,
fault geometry, units, and the color scale. Use identical limits when comparing
models, and prefer a diverging colormap centered on zero for signed slip or
residuals. Smooth-looking interpolation is not additional resolution: always
inspect the underlying patch values and resolution/uncertainty alongside an
interpolated map.

---

## `plot.slip(fault, slip_vector, **kwargs)`

Slip distribution as colored patches (rectangular or triangular).

```python
geodef.plot.slip(fault, result.slip_magnitude)

geodef.plot.slip(fault, result.slip_vector,
    ax=ax,
    components='dip',         # 'strike', 'dip', 'magnitude' (default)
    coords='fault',           # 'fault' (default) or 'geographic'
    updip_edge=True,          # black line along the up-dip edge (default)
    cmap='RdBu_r',
    vmin=-2, vmax=2,
    edgecolor='gray', linewidth=0.5,   # → PatchCollection/**kwargs
    colorbar=True,
    colorbar_label='Dip slip (m)',
    title='Coseismic slip',
)
```

By default the slip is drawn in **fault coordinates** — along-strike (x) and
along-dip (y) kilometers, with the up-dip (shallowest) edge at the top of the
axes — which is the natural frame for reading a slip distribution. A black line
marks the up-dip edge (disable with `updip_edge=False`). Pass
`coords='geographic'` to plot in local East/North kilometers instead; use this
when overlaying the slip on a map together with `plot.vectors`, `plot.insar`, or
station locations (the up-dip edge is then drawn as the surface trace). The
`coords` and `updip_edge` arguments are also available on `plot.patches`,
`plot.resolution`, and `plot.uncertainty` (defaulting to `'geographic'` and
`False` there).

Pass the named result array you want to plot, such as `result.strike_slip`,
`result.dip_slip`, `result.slip_magnitude`, `result.rake_parallel`, or
`result.rake_perpendicular`. Raw N/2N vectors remain accepted; a raw
one-component vector is plotted directly.

---

## `plot.slip_interpolated(fault, slip_vector, **kwargs)`

A smoothly interpolated slip field in map view, as an alternative to the
discrete patches of `plot.slip`. Rectangular faults are drawn with a
Gouraud-shaded `pcolormesh` over the structured grid; triangular (or
unstructured) faults use `tricontourf` over the patch centroids.

```python
geodef.plot.slip_interpolated(fault, result.slip_magnitude,
    cmap='viridis',
    levels=20,          # filled contour levels (tricontourf path)
    colorbar=True,
    title='Interpolated slip',
)
```

The `tricontourf` path needs at least three patches to triangulate.

---

## `plot.patches(fault, values, **kwargs)`

Generic per-patch scalar plot. `plot.slip`, `plot.resolution`, and `plot.uncertainty` are thin wrappers around this.

```python
geodef.plot.patches(fault, some_array, cmap='viridis', vmin=0, vmax=1)
```

---

## `plot.resolution(fault, values, **kwargs)`

Model resolution diagonal on fault patches (same interface as `plot.slip`).

```python
geodef.plot.resolution(fault, np.diag(R), cmap='viridis', vmin=0, vmax=1)
```

`diag(R)` measures how strongly each parameter is reproduced by the linearized
inverse operator, but it ignores off-diagonal smearing. Values near one are not
equivalent to small uncertainty; inspect rows of `R`, covariance, and synthetic
recovery tests for a fuller resolution assessment.

---

## `plot.uncertainty(fault, values, **kwargs)`

Per-parameter 1-sigma uncertainty on fault patches.

```python
geodef.plot.uncertainty(fault, sigma, cmap='magma_r')
```

These are standard deviations under the assumed linear model, covariance, and
regularization. They do not automatically include fault-geometry uncertainty
or systematic model error.

---

## `plot.vectors(dataset, fault, **kwargs)`

GNSS displacement/velocity vectors as quiver arrows.

```python
geodef.plot.vectors(gnss, fault,
    predicted=result.predicted[:gnss.n_obs],   # optional overlay
    scale=10,
    obs_color='black', pred_color='red',
    components='both',          # 'horizontal', 'vertical', 'both'
    legend=True,
    scale_arrow=0.5, scale_arrow_label="50 cm",
    scale_arrow_loc='lower right',
    vertical_colorbar=True,     # colorbar for the vertical dots
    vertical_size=40,           # constant dot area (points^2)
)
```

For `components='vertical'` (and the vertical part of `'both'`), the vertical
component is shown as color-coded circles (symmetric `RdBu_r`) with a colorbar;
set `vertical_colorbar=False` to suppress it. The dots are a **constant size**
(`vertical_size`) — the vertical value is encoded by color only, not marker
area. In `'both'` mode the horizontal arrows are drawn *above* the vertical dots
so they cannot be hidden.

---

## `plot.insar(dataset, fault, **kwargs)`

InSAR LOS data as colored scatter points.

```python
geodef.plot.insar(insar, fault,
    predicted=result.predicted[gnss.n_obs:],
    layout='obs_pred_res',    # 'obs', 'pred', 'residual', 'obs_pred_res'
    cmap='RdBu_r',
    vmin=-0.1, vmax=0.1,
    scatter_kwargs={'s': 2},
)
```

`layout='obs_pred_res'` returns a figure with 3 axes.

---

## `plot.fit(obs, pred, **kwargs)`

Observed vs. predicted scatter or residual histogram.

```python
geodef.plot.fit(gnss.obs, result.predicted[:gnss.n_obs])
geodef.plot.fit(gnss.obs, result.predicted[:gnss.n_obs],
    style='residual_histogram')
```

For an inversion result, the named assessment plots avoid manual slicing and
do not require the original fault or dataset objects:

```python
geodef.plot.prediction(result)                     # observed vs predicted
geodef.plot.residual(result)                       # residual histograms
geodef.plot.diagnostics(result)                    # reduced chi-squared bars
geodef.plot.diagnostics(result, metric="rms")
geodef.plot.summary(result)                        # assumptions and fit text
```

---

## `plot.fault3d(fault, **kwargs)`

3-D visualization of fault geometry.

```python
geodef.plot.fault3d(fault,
    color_by='depth',       # 'depth', 'area', 1-D array, or None
    cmap='viridis',
    show_edges=True,
    view=(30, -60),         # (elevation, azimuth) in degrees; None = matplotlib default
    aspect='equal',         # 'equal', 'auto', or a vertical-exaggeration factor
    station_locations=gnss, # optional: overlay station positions
)
```

`aspect` controls the data aspect ratio. The default `'equal'` scales all
three axes in proportion to their data ranges so the geometry is undistorted.
Shallow faults have a small depth extent relative to their horizontal
footprint, so `'equal'` can render them as a thin slab; pass a positive number
(e.g. `aspect=3`) to apply a vertical exaggeration to the depth axis, or
`'auto'` for matplotlib's default cubic box. New figures use constrained
layout and a padded colorbar so depth labels and the colorbar do not collide.

---

## `plot.map_view(fault, **kwargs)`

2-D map view of fault patches with optional station overlay.

```python
geodef.plot.map_view(fault,
    datasets=[gnss, insar],
    slip_vector=result.slip_vector,
    components='magnitude',
    cmap='YlOrRd',
    colorbar_label='Slip (m)',
    show_trace=True,
    trace_kwargs={'color': 'red', 'linewidth': 2},
    patch_kwargs={},
)
```

As with `plot.slip()`, a length-`N` one-parameter slip vector is plotted as
an amplitude directly; `components` only selects from blocked length-`2N`
strike/dip slip vectors.

---

## Composing plots

Since every function accepts `ax`, plots can be layered freely:

```python
fig, ax = plt.subplots()
geodef.plot.slip(fault, result.slip_vector, ax=ax, cmap='YlOrRd')
geodef.plot.vectors(gnss, fault, ax=ax, scale=10)
```

---

## L-curve and ABIC-curve plots

`LCurveResult` and `ABICCurveResult` have `.plot()` methods that follow the same pattern:

```python
lc = geodef.lcurve(fault, data, smoothing='laplacian', smoothing_range=(1e-2, 1e6))
ax = lc.plot()           # optimal λ annotated automatically
ax = lc.plot(ax=ax, color='navy')
```
