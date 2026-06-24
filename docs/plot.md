# `geodef.plot` — Visualization

All plot functions follow a consistent pattern:
- Accept optional `ax=None`; create a new figure if `None`
- Return the `matplotlib.axes.Axes` used
- Never call `plt.show()`
- Pass `**kwargs` through to the underlying matplotlib artist

---

## `plot.slip(fault, slip_vector, **kwargs)`

Slip distribution as colored patches (rectangular or triangular).

```python
geodef.plot.slip(fault, result.slip_vector)

geodef.plot.slip(fault, result.slip_vector,
    ax=ax,
    components='dip',         # 'strike', 'dip', 'magnitude' (default)
    cmap='RdBu_r',
    vmin=-2, vmax=2,
    edgecolor='gray', linewidth=0.5,   # → PatchCollection/**kwargs
    colorbar=True,
    colorbar_label='Dip slip (m)',
    title='Coseismic slip',
)
```

For one-parameter inversion results such as `components='rake'` or
`components='azimuth'`, `result.slip_vector` has length `N`; `plot.slip()`
plots that amplitude directly and ignores the `components` selector.

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

---

## `plot.uncertainty(fault, values, **kwargs)`

Per-parameter 1-sigma uncertainty on fault patches.

```python
geodef.plot.uncertainty(fault, sigma, cmap='magma_r')
```

---

## `plot.vectors(dataset, fault, **kwargs)`

GNSS displacement/velocity vectors as quiver arrows.

```python
geodef.plot.vectors(gnss, fault,
    predicted=result.predicted[:gnss.n_obs],   # optional overlay
    scale=10,
    obs_color='black', pred_color='red',
    components='horizontal',    # 'horizontal', 'vertical', 'both'
    legend=True,
    scale_arrow=0.5, scale_arrow_label="50 cm",
    scale_arrow_loc='lower right',
)
```

For `components='vertical'`, data is shown as color-coded circles.

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

## `plot.map(fault, **kwargs)`

2-D map view of fault patches with optional station overlay.

```python
geodef.plot.map(fault,
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
