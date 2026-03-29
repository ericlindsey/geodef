"""Visualization functions for geodef.

Every public function follows a consistent pattern:

- Accepts an optional ``ax`` parameter; creates a new figure if ``None``.
- Returns the ``matplotlib.axes.Axes`` used for plotting.
- Never calls ``plt.show()``.
- Passes ``**kwargs`` through to the underlying matplotlib artist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.pyplot as plt

    from geodef.data import DataSet, GNSS, InSAR
    from geodef.fault import Fault


# ======================================================================
# Internal helpers
# ======================================================================


def _ensure_axes(
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Return *ax* or create a new figure and axes."""
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    return ax


def _ensure_axes_3d(
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Return *ax* (must be 3D) or create a new 3D figure and axes."""
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    fig = plt.figure()
    return fig.add_subplot(111, projection="3d")


def _stations_to_local_km(
    dataset: DataSet,
    fault: Fault,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dataset station lat/lon to local Cartesian (East km, North km).

    Uses the fault's reference point as the local origin.
    """
    from geodef import transforms

    alt = np.zeros(dataset.n_stations)
    e, n, _ = transforms.geod2enu(
        dataset.lat, dataset.lon, alt,
        fault._ref_lat, fault._ref_lon, 0.0,
    )
    return e * 1e-3, n * 1e-3


def _get_patch_vertices_local(fault: Fault) -> list[np.ndarray]:
    """Compute 2-D patch vertices in local Cartesian (East km, North km).

    Args:
        fault: Fault geometry.

    Returns:
        List of arrays, each (n_corners, 2). Rectangular patches have
        4 corners; triangular patches have 3.
    """
    from geodef import transforms

    ref_lat = fault._ref_lat
    ref_lon = fault._ref_lon

    if fault.engine == "okada":
        sin_dip = np.sin(np.radians(fault._dip))
        cos_dip = np.cos(np.radians(fault._dip))
        sin_str = np.sin(np.radians(fault._strike))
        cos_str = np.cos(np.radians(fault._strike))

        half_L = fault._length / 2
        half_W = fault._width / 2

        # Corner offsets in ENU (meters) relative to patch center
        # Order: top-left, top-right, bottom-right, bottom-left
        # "top" = updip (shallower)
        e_off = np.column_stack([
            -half_L * sin_str + half_W * cos_dip * cos_str,
            +half_L * sin_str + half_W * cos_dip * cos_str,
            +half_L * sin_str - half_W * cos_dip * cos_str,
            -half_L * sin_str - half_W * cos_dip * cos_str,
        ])
        n_off = np.column_stack([
            -half_L * cos_str - half_W * cos_dip * sin_str,
            +half_L * cos_str - half_W * cos_dip * sin_str,
            +half_L * cos_str + half_W * cos_dip * sin_str,
            -half_L * cos_str + half_W * cos_dip * sin_str,
        ])

        # Patch centers in local ENU (meters)
        alt = np.zeros(fault.n_patches)
        ce, cn, _ = transforms.geod2enu(
            fault._lat, fault._lon, alt, ref_lat, ref_lon, 0.0,
        )

        verts = []
        for i in range(fault.n_patches):
            corners = np.column_stack([
                (ce[i] + e_off[i]) * 1e-3,
                (cn[i] + n_off[i]) * 1e-3,
            ])
            verts.append(corners)
        return verts

    # Triangular: _vertices is (N, 3, 3) in local ENU meters [e, n, u]
    tri_verts = fault._vertices
    verts = []
    for i in range(fault.n_patches):
        corners = tri_verts[i, :, :2] * 1e-3  # (3, 2) east/north in km
        verts.append(corners)
    return verts


def _get_patch_vertices_3d(fault: Fault) -> list[np.ndarray]:
    """Compute 3-D patch vertices in local Cartesian (East km, North km, Depth km).

    Depth is positive downward for display.

    Returns:
        List of arrays, each (n_corners, 3).
    """
    from geodef import transforms

    ref_lat = fault._ref_lat
    ref_lon = fault._ref_lon

    if fault.engine == "okada":
        sin_dip = np.sin(np.radians(fault._dip))
        cos_dip = np.cos(np.radians(fault._dip))
        sin_str = np.sin(np.radians(fault._strike))
        cos_str = np.cos(np.radians(fault._strike))

        half_L = fault._length / 2
        half_W = fault._width / 2

        e_off = np.column_stack([
            -half_L * sin_str + half_W * cos_dip * cos_str,
            +half_L * sin_str + half_W * cos_dip * cos_str,
            +half_L * sin_str - half_W * cos_dip * cos_str,
            -half_L * sin_str - half_W * cos_dip * cos_str,
        ])
        n_off = np.column_stack([
            -half_L * cos_str - half_W * cos_dip * sin_str,
            +half_L * cos_str - half_W * cos_dip * sin_str,
            +half_L * cos_str + half_W * cos_dip * sin_str,
            -half_L * cos_str + half_W * cos_dip * sin_str,
        ])
        # Depth offsets (positive = deeper)
        d_off = np.column_stack([
            -half_W * sin_dip,
            -half_W * sin_dip,
            +half_W * sin_dip,
            +half_W * sin_dip,
        ])

        alt = np.zeros(fault.n_patches)
        ce, cn, _ = transforms.geod2enu(
            fault._lat, fault._lon, alt, ref_lat, ref_lon, 0.0,
        )

        verts = []
        for i in range(fault.n_patches):
            corners = np.column_stack([
                (ce[i] + e_off[i]) * 1e-3,
                (cn[i] + n_off[i]) * 1e-3,
                (fault._depth[i] + d_off[i]) * 1e-3,
            ])
            verts.append(corners)
        return verts

    # Triangular
    tri_verts = fault._vertices  # (N, 3, 3) [e, n, u]
    verts = []
    for i in range(fault.n_patches):
        v = tri_verts[i] * 1e-3  # (3, 3) in km
        # Convert up to depth: depth_km = -up_km
        corners = np.column_stack([v[:, 0], v[:, 1], -v[:, 2]])
        verts.append(corners)
    return verts


def _get_slip_component(
    slip: np.ndarray,
    n_patches: int,
    component: str,
) -> np.ndarray:
    """Extract a scalar per patch from a blocked slip vector.

    Args:
        slip: Blocked slip vector ``[ss_0..ss_N, ds_0..ds_N]``, length 2*N.
        n_patches: Number of fault patches (N).
        component: One of ``'strike'``, ``'dip'``, ``'magnitude'``.

    Returns:
        Array of shape (N,).

    Raises:
        ValueError: If *component* is invalid or *slip* has wrong length.
    """
    if slip.shape[0] != 2 * n_patches:
        raise ValueError(
            f"slip length {slip.shape[0]} does not match "
            f"2 * n_patches = {2 * n_patches}"
        )
    ss = slip[:n_patches]
    ds = slip[n_patches:]
    if component == "strike":
        return ss
    if component == "dip":
        return ds
    if component == "magnitude":
        return np.sqrt(ss**2 + ds**2)
    raise ValueError(
        f"Unknown component {component!r}. Use 'strike', 'dip', or 'magnitude'."
    )


def _plot_patch_scalar(
    fault: Fault,
    values: np.ndarray,
    *,
    ax: matplotlib.axes.Axes | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    colorbar_kwargs: dict | None = None,
    title: str | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot a scalar value on each fault patch as colored polygons.

    This is the shared implementation behind ``slip()``, ``resolution()``,
    ``uncertainty()``, and ``patches()``.
    """
    from matplotlib.collections import PolyCollection

    ax = _ensure_axes(ax)
    verts = _get_patch_vertices_local(fault)

    defaults = {"edgecolor": "face", "linewidth": 0.5}
    defaults.update(kwargs)

    pc = PolyCollection(verts, **defaults)
    pc.set_array(values)
    pc.set_cmap(cmap)
    if vmin is not None or vmax is not None:
        pc.set_clim(vmin, vmax)

    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")

    if title is not None:
        ax.set_title(title)

    if colorbar:
        cb_kw = colorbar_kwargs or {}
        cb = ax.figure.colorbar(pc, ax=ax, **cb_kw)
        if colorbar_label is not None:
            cb.set_label(colorbar_label)

    return ax


def _get_surface_trace(fault: Fault) -> np.ndarray | None:
    """Compute the surface trace of a fault as an ordered line.

    The surface trace is the updip (shallowest) edge of the shallowest
    row of patches. For a structured rectangular grid, this is the row
    with the smallest patch-center depth. For unstructured meshes, we
    find vertices closest to the surface.

    Returns:
        Array of shape (M, 2) with (east_km, north_km), or None if no
        trace can be determined.
    """
    verts_3d = _get_patch_vertices_3d(fault)
    verts_2d = _get_patch_vertices_local(fault)

    if fault.engine == "okada":
        # For rectangular patches, corners 0 and 1 are the updip edge.
        # Find the shallowest patches (minimum depth at center).
        if fault.grid_shape is not None:
            nL, nW = fault.grid_shape
            # Patches are ordered (dip outer, strike inner).
            # Find which dip row has the shallowest depth.
            row_depths = np.array([
                np.mean(fault._depth[j * nL:(j + 1) * nL])
                for j in range(nW)
            ])
            shallow_row = np.argmin(row_depths)
            shallow_indices = list(range(shallow_row * nL,
                                         (shallow_row + 1) * nL))
        else:
            # Unstructured: pick patches in the shallowest depth quartile
            depth_threshold = np.percentile(fault._depth, 25)
            shallow_indices = np.where(
                fault._depth <= depth_threshold
            )[0].tolist()

        # Collect updip edge endpoints (corners 0 and 1)
        edge_points = []
        for idx in shallow_indices:
            edge_points.append(verts_2d[idx][0])  # top-left
            edge_points.append(verts_2d[idx][1])  # top-right
        if not edge_points:
            return None
        edge_points = np.array(edge_points)

        # Remove near-duplicates, then sort along the trace.
        # Use PCA-like approach: project onto the dominant direction.
        _, unique_idx = np.unique(
            np.round(edge_points, 5), axis=0, return_index=True
        )
        pts = edge_points[np.sort(unique_idx)]

        # Sort by projecting onto the along-strike direction
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        # Use SVD to find the principal direction
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        projections = centered @ Vt[0]
        order = np.argsort(projections)
        return pts[order]

    if fault.engine == "tri":
        # For triangular faults, find vertices closest to the surface
        all_3d = np.vstack(verts_3d)
        min_depth = np.min(all_3d[:, 2])
        depth_range = np.max(all_3d[:, 2]) - min_depth
        threshold = min_depth + depth_range * 0.1 if depth_range > 0 else \
            min_depth + 1.0

        top_points = []
        for v2d, v3d in zip(verts_2d, verts_3d):
            for j in range(len(v3d)):
                if v3d[j, 2] <= threshold:
                    top_points.append(v2d[j])
        if not top_points:
            return None
        top_points = np.array(top_points)
        _, unique_idx = np.unique(
            np.round(top_points, 5), axis=0, return_index=True
        )
        pts = top_points[np.sort(unique_idx)]
        # Sort by projecting onto principal direction
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        if len(pts) >= 2:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            projections = centered @ Vt[0]
            order = np.argsort(projections)
            return pts[order]
        return pts

    return None


# ======================================================================
# Public API — Fault patch plots
# ======================================================================


def patches(
    fault: Fault,
    values: np.ndarray,
    *,
    ax: matplotlib.axes.Axes | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    colorbar_kwargs: dict | None = None,
    title: str | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot an arbitrary scalar quantity on fault patches.

    This is the general-purpose patch-coloring function. Use it for any
    per-patch quantity (depth, area, coupling ratio, etc.).

    Args:
        fault: Fault geometry (rectangular or triangular).
        values: Scalar per patch, shape (N,) where N is the number of
            fault patches.
        ax: Axes to plot on. Creates a new figure if ``None``.
        cmap: Matplotlib colormap name.
        vmin: Minimum color limit.
        vmax: Maximum color limit.
        colorbar: Whether to add a colorbar.
        colorbar_label: Colorbar label.
        colorbar_kwargs: Extra kwargs passed to ``fig.colorbar()``.
        title: Axes title.
        **kwargs: Passed to ``PolyCollection`` (e.g. ``edgecolor``,
            ``linewidth``).

    Returns:
        The axes used for plotting.
    """
    return _plot_patch_scalar(
        fault, values,
        ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar=colorbar, colorbar_label=colorbar_label,
        colorbar_kwargs=colorbar_kwargs, title=title,
        **kwargs,
    )


def slip(
    fault: Fault,
    slip_vector: np.ndarray,
    *,
    ax: matplotlib.axes.Axes | None = None,
    component: str = "magnitude",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    colorbar_kwargs: dict | None = None,
    title: str | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot fault slip distribution as colored patches.

    Args:
        fault: Fault geometry (rectangular or triangular).
        slip_vector: Blocked slip ``[ss_0..ss_N, ds_0..ds_N]``, length 2*N.
        ax: Axes to plot on. Creates a new figure if ``None``.
        component: Which slip component to display. One of ``'strike'``,
            ``'dip'``, or ``'magnitude'`` (default).
        cmap: Matplotlib colormap name.
        vmin: Minimum color limit.
        vmax: Maximum color limit.
        colorbar: Whether to add a colorbar.
        colorbar_label: Colorbar label. Auto-generated if ``None``.
        colorbar_kwargs: Extra kwargs passed to ``fig.colorbar()``.
        title: Axes title.
        **kwargs: Passed to ``PolyCollection`` (e.g. ``edgecolor``,
            ``linewidth``).

    Returns:
        The axes used for plotting.
    """
    values = _get_slip_component(slip_vector, fault.n_patches, component)
    if colorbar_label is None:
        labels = {
            "strike": "Strike-slip (m)",
            "dip": "Dip-slip (m)",
            "magnitude": "Slip magnitude (m)",
        }
        colorbar_label = labels.get(component, "Slip (m)")
    return _plot_patch_scalar(
        fault, values,
        ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar=colorbar, colorbar_label=colorbar_label,
        colorbar_kwargs=colorbar_kwargs, title=title,
        **kwargs,
    )


def resolution(
    fault: Fault,
    values: np.ndarray,
    *,
    ax: matplotlib.axes.Axes | None = None,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    colorbar: bool = True,
    colorbar_label: str = "Resolution",
    colorbar_kwargs: dict | None = None,
    title: str | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot resolution matrix diagonal on fault patches.

    Args:
        fault: Fault geometry.
        values: Resolution diagonal, shape (N,).
        ax: Axes to plot on. Creates a new figure if ``None``.
        cmap: Matplotlib colormap name.
        vmin: Minimum color limit.
        vmax: Maximum color limit.
        colorbar: Whether to add a colorbar.
        colorbar_label: Colorbar label.
        colorbar_kwargs: Extra kwargs passed to ``fig.colorbar()``.
        title: Axes title.
        **kwargs: Passed to ``PolyCollection``.

    Returns:
        The axes used for plotting.
    """
    return _plot_patch_scalar(
        fault, values,
        ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar=colorbar, colorbar_label=colorbar_label,
        colorbar_kwargs=colorbar_kwargs, title=title,
        **kwargs,
    )


def uncertainty(
    fault: Fault,
    values: np.ndarray,
    *,
    ax: matplotlib.axes.Axes | None = None,
    cmap: str = "magma_r",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str = "1-sigma uncertainty (m)",
    colorbar_kwargs: dict | None = None,
    title: str | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot model uncertainty on fault patches.

    Args:
        fault: Fault geometry.
        values: Uncertainty per patch, shape (N,).
        ax: Axes to plot on. Creates a new figure if ``None``.
        cmap: Matplotlib colormap name.
        vmin: Minimum color limit.
        vmax: Maximum color limit.
        colorbar: Whether to add a colorbar.
        colorbar_label: Colorbar label.
        colorbar_kwargs: Extra kwargs passed to ``fig.colorbar()``.
        title: Axes title.
        **kwargs: Passed to ``PolyCollection``.

    Returns:
        The axes used for plotting.
    """
    return _plot_patch_scalar(
        fault, values,
        ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar=colorbar, colorbar_label=colorbar_label,
        colorbar_kwargs=colorbar_kwargs, title=title,
        **kwargs,
    )


# ======================================================================
# Public API — Data observation plots
# ======================================================================


def vectors(
    dataset: GNSS,
    fault: Fault,
    *,
    predicted: np.ndarray | None = None,
    ax: matplotlib.axes.Axes | None = None,
    scale: float = 1.0,
    components: str = "horizontal",
    obs_color: str = "black",
    pred_color: str = "red",
    ellipses: bool = True,
    ellipse_kwargs: dict | None = None,
    legend: bool = False,
    scale_arrow: float | None = None,
    scale_arrow_label: str | None = None,
    scale_arrow_loc: str = "lower right",
    quiver_kwargs: dict | None = None,
    vertical_colorbar: bool = True,
    vertical_colorbar_label: str = "Vertical displacement",
    title: str | None = None,
) -> matplotlib.axes.Axes:
    """Plot GNSS displacement/velocity vectors.

    Args:
        dataset: GNSS dataset with observed vectors.
        fault: Fault geometry (used as local coordinate reference).
        predicted: Predicted observation vector matching ``dataset.obs``
            layout, or ``None`` to plot only observed.
        ax: Axes to plot on. Creates a new figure if ``None``.
        scale: Scale factor for horizontal vector lengths. Vectors are
            plotted in data coordinates (km), so set this to convert
            your displacement units into a visible length in km.
            For vertical-only mode, ``scale`` controls the dot size
            scaling (marker area is proportional to ``|value| * scale``).
        components: ``'horizontal'``, ``'vertical'``, or ``'both'``.
        obs_color: Color for observed vectors.
        pred_color: Color for predicted vectors.
        ellipses: Whether to draw uncertainty ellipses (horizontal only).
            Ellipses are drawn at the arrow tip (standard geodetic convention).
        ellipse_kwargs: Extra kwargs passed to ``Ellipse`` patches.
        legend: Whether to add a legend. When ``scale_arrow`` is set, the
            legend includes a reference arrow; otherwise it uses standard
            matplotlib entries.
        scale_arrow: If set, the *data-unit* magnitude of a reference arrow
            drawn in the legend (e.g. ``scale_arrow=0.01`` for a "10 mm"
            arrow when data are in meters). The arrow is plotted at
            ``scale_arrow * scale`` km length.
        scale_arrow_label: Label for the reference arrow. Auto-generated
            as ``"{scale_arrow} observed"`` if ``None``.
        scale_arrow_loc: Position of the scale arrow legend. One of
            ``'lower right'``, ``'lower left'``, ``'upper right'``,
            ``'upper left'``.
        quiver_kwargs: Extra kwargs passed to ``ax.quiver()``.
        vertical_colorbar: Whether to show a colorbar for the vertical
            component when ``components='vertical'``.
        vertical_colorbar_label: Label for the vertical colorbar.
        title: Axes title.

    Returns:
        The axes used for plotting.

    Raises:
        ValueError: If ``components='vertical'`` but dataset has no
            vertical component.
    """
    from matplotlib.patches import Ellipse

    ax = _ensure_axes(ax)
    x_km, y_km = _stations_to_local_km(dataset, fault)
    n = dataset.n_stations
    has_vert = dataset.components == "enu"

    if components == "vertical" and not has_vert:
        raise ValueError(
            "Vertical component requested but dataset is horizontal-only."
        )

    qkw = {"angles": "xy", "scale_units": "xy", "scale": 1}
    if quiver_kwargs:
        qkw.update(quiver_kwargs)

    if components in ("horizontal", "both"):
        ve = dataset._ve * scale
        vn = dataset._vn * scale
        ax.quiver(x_km, y_km, ve, vn, color=obs_color, label="_nolegend_",
                  **qkw)

        if predicted is not None:
            if has_vert:
                pe = predicted[0::3] * scale
                pn = predicted[1::3] * scale
            else:
                pe = predicted[0::2] * scale
                pn = predicted[1::2] * scale
            ax.quiver(x_km, y_km, pe, pn, color=pred_color,
                      label="_nolegend_", **qkw)

        if ellipses:
            ekw = {"facecolor": "none", "edgecolor": obs_color,
                   "linewidth": 0.5, "alpha": 0.4}
            if ellipse_kwargs:
                ekw.update(ellipse_kwargs)
            for i in range(n):
                w = dataset._se[i] * scale * 2  # full axis = 2 * sigma
                h = dataset._sn[i] * scale * 2
                # Place ellipse at the arrow tip (geodetic convention)
                tip_x = x_km[i] + ve[i]
                tip_y = y_km[i] + vn[i]
                ell = Ellipse((tip_x, tip_y), w, h, **ekw)
                ax.add_patch(ell)

    if components == "vertical" or (components == "both" and has_vert):
        vu = dataset._vu  # raw values — don't scale the color
        # Scale controls dot size, not color
        base_size = 30
        sizes = np.abs(vu) / np.max(np.abs(vu)) * base_size * scale \
            if np.max(np.abs(vu)) > 0 else np.full(n, base_size)
        # Clamp minimum size so dots are always visible
        sizes = np.clip(sizes, base_size * 0.2, None)

        sc = ax.scatter(x_km, y_km, c=vu, s=sizes, cmap="RdBu_r",
                        edgecolors="k", linewidths=0.5, zorder=5,
                        label="Vertical (obs)" if components == "vertical"
                        else None)
        if components == "vertical" and vertical_colorbar:
            ax.figure.colorbar(sc, ax=ax, label=vertical_colorbar_label)

        if predicted is not None and has_vert:
            pu = predicted[2::3]
            ax.scatter(x_km, y_km, c=pu, s=sizes, cmap="RdBu_r",
                       edgecolors=pred_color, linewidths=1.0,
                       facecolors="none", marker="o", zorder=6)

    ax.set_aspect("equal")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    if title is not None:
        ax.set_title(title)

    if legend:
        if scale_arrow is not None:
            _add_scale_arrow_legend(
                ax, scale_arrow, scale,
                obs_color=obs_color,
                pred_color=pred_color if predicted is not None else None,
                label=scale_arrow_label,
                loc=scale_arrow_loc,
                quiver_kwargs=quiver_kwargs,
            )
        else:
            # Fallback: standard legend with proxy artists
            from matplotlib.lines import Line2D

            handles = [
                Line2D([0], [0], color=obs_color, linewidth=2,
                       label="Observed"),
            ]
            if predicted is not None:
                handles.append(
                    Line2D([0], [0], color=pred_color, linewidth=2,
                           label="Predicted"),
                )
            ax.legend(handles=handles)

    ax.autoscale_view()
    return ax


def _add_scale_arrow_legend(
    ax: matplotlib.axes.Axes,
    scale_arrow: float,
    scale: float,
    *,
    obs_color: str = "black",
    pred_color: str | None = None,
    label: str | None = None,
    loc: str = "lower right",
    quiver_kwargs: dict | None = None,
) -> None:
    """Add a scale-bar arrow legend to a vector plot.

    Uses ``ax.quiver()`` so the reference arrow matches the data arrows
    exactly (same head shape, line width, etc.).
    """
    if label is None:
        label = f"{scale_arrow} observed"

    arrow_km = scale_arrow * scale

    # Determine anchor position from loc
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    if "right" in loc:
        bx = xlim[1] - x_range * 0.05
        ha = "right"
        text_x = bx
    else:
        bx = xlim[0] + x_range * 0.05 + arrow_km
        ha = "left"
        text_x = bx

    if "upper" in loc:
        by = ylim[1] - y_range * 0.08
        text_dy = y_range * 0.02
        row_dy = -y_range * 0.05
    else:
        by = ylim[0] + y_range * 0.08
        text_dy = y_range * 0.02
        row_dy = y_range * 0.05

    # Build quiver kwargs that match the data arrows
    qkw = {"angles": "xy", "scale_units": "xy", "scale": 1}
    if quiver_kwargs:
        qkw.update(quiver_kwargs)

    ax.quiver(bx, by, -arrow_km, 0, color=obs_color, clip_on=False, **qkw)
    ax.text(text_x, by + text_dy, label, ha=ha, va="bottom",
            fontsize=8, color=obs_color)

    if pred_color is not None:
        by2 = by + row_dy
        pred_label = (
            label.replace("observed", "predicted") if "observed" in label
            else f"{scale_arrow} predicted"
        )
        ax.quiver(bx, by2, -arrow_km, 0, color=pred_color, clip_on=False,
                  **qkw)
        ax.text(text_x, by2 + text_dy, pred_label, ha=ha, va="bottom",
                fontsize=8, color=pred_color)


def insar(
    dataset: InSAR,
    fault: Fault,
    *,
    predicted: np.ndarray | None = None,
    ax: matplotlib.axes.Axes | None = None,
    layout: str = "obs",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = "LOS displacement",
    scatter_kwargs: dict | None = None,
    title: str | None = None,
) -> matplotlib.axes.Axes | np.ndarray:
    """Plot InSAR LOS data as colored scatter points.

    Args:
        dataset: InSAR dataset.
        fault: Fault geometry (used as local coordinate reference).
        predicted: Predicted LOS values, shape ``(n_stations,)``.
        ax: Axes to plot on (single-panel layouts only). Creates a new
            figure if ``None``.
        layout: Panel layout. One of:

            - ``'obs'`` — single panel with observations (default).
            - ``'pred'`` — single panel with predictions (requires *predicted*).
            - ``'residual'`` — single panel with obs minus pred.
            - ``'obs_pred_res'`` — three side-by-side panels.
        cmap: Matplotlib colormap name.
        vmin: Minimum color limit. If ``None``, auto-scaled symmetrically.
        vmax: Maximum color limit.
        colorbar: Whether to add a colorbar.
        colorbar_label: Colorbar label.
        scatter_kwargs: Extra kwargs passed to ``ax.scatter()``.
        title: Axes title (single-panel) or ignored for multi-panel.

    Returns:
        Single ``Axes`` for single-panel layouts, or ``numpy`` array of
        3 ``Axes`` for ``'obs_pred_res'``.

    Raises:
        ValueError: If *predicted* is required but not provided.
    """
    import matplotlib.pyplot as plt

    x_km, y_km = _stations_to_local_km(dataset, fault)
    skw = {"s": 8, "cmap": cmap}
    if scatter_kwargs:
        skw.update(scatter_kwargs)
    # Don't pass cmap twice
    skw.setdefault("cmap", cmap)

    def _auto_clim(data: np.ndarray):
        if vmin is not None and vmax is not None:
            return vmin, vmax
        absmax = np.max(np.abs(data))
        return (vmin if vmin is not None else -absmax,
                vmax if vmax is not None else absmax)

    def _scatter_panel(ax_panel, data, panel_title, show_ylabel=True):
        lo, hi = _auto_clim(data)
        sc = ax_panel.scatter(x_km, y_km, c=data, vmin=lo, vmax=hi, **skw)
        ax_panel.set_aspect("equal")
        ax_panel.set_xlabel("East (km)")
        if show_ylabel:
            ax_panel.set_ylabel("North (km)")
        ax_panel.set_title(panel_title)
        return sc

    if layout == "obs_pred_res":
        if predicted is None:
            raise ValueError("predicted is required for 'obs_pred_res' layout")
        residual = dataset.obs - predicted
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        sc0 = _scatter_panel(axes[0], dataset.obs, "Observed")
        sc1 = _scatter_panel(axes[1], predicted, "Predicted",
                             show_ylabel=False)
        sc2 = _scatter_panel(axes[2], residual, "Residual",
                             show_ylabel=False)
        if colorbar:
            fig.colorbar(sc0, ax=axes[:2], label=colorbar_label, shrink=0.8)
            fig.colorbar(sc2, ax=axes[2], label="Residual", shrink=0.8)
        return axes

    # Single-panel layouts
    if layout in ("pred", "residual") and predicted is None:
        raise ValueError(f"predicted is required for {layout!r} layout")

    if layout == "obs":
        data = dataset.obs
        panel_title = title or "Observed"
    elif layout == "pred":
        data = predicted
        panel_title = title or "Predicted"
    elif layout == "residual":
        data = dataset.obs - predicted
        panel_title = title or "Residual"
    else:
        raise ValueError(
            f"Unknown layout {layout!r}. "
            "Use 'obs', 'pred', 'residual', or 'obs_pred_res'."
        )

    ax = _ensure_axes(ax)
    sc = _scatter_panel(ax, data, panel_title)
    if colorbar:
        ax.figure.colorbar(sc, ax=ax, label=colorbar_label)
    return ax


def fit(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    ax: matplotlib.axes.Axes | None = None,
    style: str = "scatter",
    scatter_kwargs: dict | None = None,
    title: str | None = None,
) -> matplotlib.axes.Axes:
    """Plot observed vs. predicted values.

    Args:
        observed: Observed data vector.
        predicted: Predicted data vector (same length).
        ax: Axes to plot on. Creates a new figure if ``None``.
        style: ``'scatter'`` for obs-vs-pred with 1:1 line, or
            ``'residual_histogram'`` for a histogram of residuals.
        scatter_kwargs: Extra kwargs passed to ``ax.scatter()`` (scatter
            style only).
        title: Axes title.

    Returns:
        The axes used for plotting.

    Raises:
        ValueError: If *style* is invalid.
    """
    ax = _ensure_axes(ax)

    if style == "scatter":
        skw = {"s": 10, "alpha": 0.7, "edgecolors": "none"}
        if scatter_kwargs:
            skw.update(scatter_kwargs)
        ax.scatter(observed, predicted, **skw)
        lo = min(np.min(observed), np.min(predicted))
        hi = max(np.max(observed), np.max(predicted))
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", linewidth=0.8, label="1:1")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title(title or "Observed vs. Predicted")

    elif style == "residual_histogram":
        residuals = observed - predicted
        ax.hist(residuals, bins="auto", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Residual (obs - pred)")
        ax.set_ylabel("Count")
        ax.set_title(title or "Residual histogram")

    else:
        raise ValueError(
            f"Unknown style {style!r}. Use 'scatter' or 'residual_histogram'."
        )

    return ax


# ======================================================================
# Public API — Fault geometry visualization
# ======================================================================


def fault3d(
    fault: Fault,
    *,
    ax: matplotlib.axes.Axes | None = None,
    color_by: str | np.ndarray | None = "depth",
    cmap: str = "viridis",
    show_edges: bool = True,
    station_locations: DataSet | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    view: tuple[float, float] | None = None,
    title: str | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """3-D visualization of fault geometry.

    Args:
        fault: Fault geometry.
        ax: 3-D axes to plot on. Creates a new 3-D figure if ``None``.
        color_by: How to color patches. ``'depth'`` uses patch center
            depth; ``'area'`` uses patch area; a 1-D array of length
            ``n_patches`` uses those values directly; ``None`` uses a
            uniform color.
        cmap: Matplotlib colormap name.
        show_edges: Whether to draw patch edges.
        station_locations: Optional dataset whose station locations are
            overlaid as scatter points at the surface.
        colorbar: Whether to add a colorbar (ignored when *color_by* is
            ``None``).
        colorbar_label: Colorbar label. Auto-generated if ``None``.
        view: ``(elevation, azimuth)`` in degrees for the 3-D view angle.
            If ``None``, uses matplotlib's default (30, -60).
        title: Axes title.
        **kwargs: Passed to ``Poly3DCollection`` (e.g. ``alpha``).

    Returns:
        The 3-D axes used for plotting.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax = _ensure_axes_3d(ax)
    verts_3d = _get_patch_vertices_3d(fault)

    # Determine face colors
    if color_by is None:
        face_values = None
    elif isinstance(color_by, str) and color_by == "depth":
        face_values = fault._depth * 1e-3  # km
        if colorbar_label is None:
            colorbar_label = "Depth (km)"
    elif isinstance(color_by, str) and color_by == "area":
        face_values = fault.areas * 1e-6  # km^2
        if colorbar_label is None:
            colorbar_label = "Area (km$^2$)"
    else:
        face_values = np.asarray(color_by)
        if face_values.shape != (fault.n_patches,):
            raise ValueError(
                f"color_by array must have length {fault.n_patches}, "
                f"got {face_values.shape[0]}"
            )

    defaults = {"edgecolor": "gray" if show_edges else "none",
                "linewidth": 0.5 if show_edges else 0,
                "alpha": 0.8}
    defaults.update(kwargs)

    pc = Poly3DCollection(verts_3d, **defaults)

    if face_values is not None:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        norm = mcolors.Normalize(vmin=np.min(face_values),
                                  vmax=np.max(face_values))
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        pc.set_facecolor(mapper.to_rgba(face_values))

        if colorbar:
            ax.figure.colorbar(mapper, ax=ax, label=colorbar_label,
                                shrink=0.6)

    ax.add_collection3d(pc)

    # Station locations — plot BEFORE setting limits so they're included.
    # Use computed_zorder=False to force manual zorder, which is the best
    # available workaround for matplotlib's 3D rendering order issue.
    if station_locations is not None:
        ax.computed_zorder = False
        pc.set_zorder(1)
        sx, sy = _stations_to_local_km(station_locations, fault)
        ax.scatter(sx, sy, np.zeros_like(sx), c="red", s=30,
                   marker="^", zorder=10, depthshade=False,
                   label="Stations")

    # Set axis limits from vertices
    all_verts = np.vstack(verts_3d)
    pad = 0.05
    for dim, setter in enumerate(
        [ax.set_xlim, ax.set_ylim, ax.set_zlim]
    ):
        lo, hi = all_verts[:, dim].min(), all_verts[:, dim].max()
        margin = (hi - lo) * pad or 1.0
        setter(lo - margin, hi + margin)

    # Invert z so depth increases downward
    if ax.get_zlim()[0] < ax.get_zlim()[1]:
        ax.invert_zaxis()

    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_zlabel("Depth (km)")

    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])

    if title is not None:
        ax.set_title(title)

    return ax


def map(
    fault: Fault,
    *,
    datasets: DataSet | list[DataSet] | None = None,
    values: np.ndarray | None = None,
    slip_vector: np.ndarray | None = None,
    component: str = "magnitude",
    ax: matplotlib.axes.Axes | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    show_patches: bool = True,
    show_trace: bool = False,
    patch_kwargs: dict | None = None,
    trace_kwargs: dict | None = None,
    title: str | None = None,
) -> matplotlib.axes.Axes:
    """2-D map view of fault geometry and optional station locations.

    Patches can be colored by a per-patch scalar using either ``values``
    (a length-*n_patches* array) or ``slip_vector`` (a blocked
    ``[ss | ds]`` vector, which is decomposed via ``component``).
    When neither is provided, patches are drawn with a uniform fill.

    Args:
        fault: Fault geometry.
        datasets: One or more datasets whose station locations are
            overlaid on the map.
        values: Per-patch scalar array (length *n_patches*) to color
            the patches by. Mutually exclusive with ``slip_vector``.
        slip_vector: Blocked ``[ss | ds]`` slip vector (length
            *2 × n_patches*). Decomposed via ``component``.
        component: Which slip component to extract when using
            ``slip_vector``. One of ``'magnitude'``, ``'strike'``,
            ``'dip'`` (default ``'magnitude'``).
        ax: Axes to plot on. Creates a new figure if ``None``.
        cmap: Colormap name (used when ``values`` or ``slip_vector``
            is provided).
        vmin: Colormap minimum.
        vmax: Colormap maximum.
        colorbar: Whether to show a colorbar (default ``True``; only
            applies when patches are colored by values).
        colorbar_label: Label for the colorbar.
        show_patches: Whether to draw fault patch outlines.
        show_trace: Whether to draw the fault's surface trace (updip
            edge of the shallowest row of patches).
        patch_kwargs: Extra kwargs for ``PolyCollection`` of fault patches.
        trace_kwargs: Extra kwargs for the surface trace line.
        title: Axes title.

    Returns:
        The axes used for plotting.
    """
    from matplotlib.collections import PolyCollection

    ax = _ensure_axes(ax)

    # Determine per-patch color values
    color_values = None
    if values is not None and slip_vector is not None:
        raise ValueError("Provide either 'values' or 'slip_vector', not both.")
    if values is not None:
        color_values = np.asarray(values)
    elif slip_vector is not None:
        color_values = _get_slip_component(
            slip_vector, fault.n_patches, component
        )

    if show_patches:
        verts = _get_patch_vertices_local(fault)
        pkw: dict = {"edgecolor": "black", "linewidth": 0.5}
        if color_values is None:
            pkw["facecolor"] = "lightgray"
            pkw["alpha"] = 0.5
        if patch_kwargs:
            pkw.update(patch_kwargs)
        pc = PolyCollection(verts, **pkw)
        if color_values is not None:
            pc.set_array(color_values)
            pc.set_cmap(cmap)
            if vmin is not None or vmax is not None:
                pc.set_clim(vmin, vmax)
        ax.add_collection(pc)

        if color_values is not None and colorbar:
            cb = ax.figure.colorbar(pc, ax=ax)
            if colorbar_label is not None:
                cb.set_label(colorbar_label)

    if show_trace:
        trace = _get_surface_trace(fault)
        if trace is not None:
            tkw = {"color": "red", "linewidth": 2, "zorder": 5}
            if trace_kwargs:
                tkw.update(trace_kwargs)
            ax.plot(trace[:, 0], trace[:, 1], **tkw)

    # Overlay dataset station locations
    if datasets is not None:
        if not isinstance(datasets, list):
            datasets = [datasets]
        markers = ["^", "o", "s", "D", "v"]
        colors = ["C0", "C1", "C2", "C3", "C4"]
        for i, ds in enumerate(datasets):
            sx, sy = _stations_to_local_km(ds, fault)
            mk = markers[i % len(markers)]
            cl = colors[i % len(colors)]
            label = type(ds).__name__
            ax.scatter(sx, sy, marker=mk, color=cl, s=30,
                       zorder=10, label=label)

    ax.set_aspect("equal")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.autoscale_view()
    if title is not None:
        ax.set_title(title)

    return ax
