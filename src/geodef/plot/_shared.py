"""Shared axes, vertex, and scalar-patch helpers for geodef.plot.

Private submodule of :mod:`geodef.plot`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    from mpl_toolkits.mplot3d import Axes3D

    from geodef.data import DataSet
    from geodef.fault import Fault


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
    ax: Axes3D | None = None,
) -> Axes3D:
    """Return *ax* (must be 3D) or create a new 3D figure and axes.

    New figures use a roomier default size and constrained layout so that
    axis labels, tick labels, and a colorbar do not collide — important
    for shallow faults whose depth extent is small relative to their
    horizontal footprint.
    """
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6), layout="constrained")
    return fig.add_subplot(111, projection="3d")


def _apply_3d_aspect(
    ax: Axes3D,
    aspect: str | float,
    zoom: float = 0.85,
) -> None:
    """Set the data aspect ratio of a 3-D fault axes.

    Args:
        ax: 3-D axes whose limits have already been set.
        aspect: ``'equal'`` for true 1:1:1 scaling (the box is sized in
            proportion to the data ranges); ``'auto'`` for matplotlib's
            default cubic box; or a positive number giving a vertical
            exaggeration factor applied to the depth axis relative to
            equal horizontal scaling (``1.0`` is identical to
            ``'equal'``, ``3.0`` stretches depth three-fold).
        zoom: Box zoom factor passed to ``set_box_aspect``; values below
            ``1`` leave a margin around the box so labels are not clipped.

    Raises:
        ValueError: If *aspect* is neither ``'equal'``/``'auto'`` nor a
            positive number.
    """
    if aspect == "auto":
        return
    if aspect == "equal":
        vertical_exaggeration = 1.0
    elif isinstance(aspect, (int, float)) and not isinstance(aspect, bool):
        vertical_exaggeration = float(aspect)
        if vertical_exaggeration <= 0:
            raise ValueError(f"aspect must be a positive number, got {aspect!r}")
    else:
        raise ValueError(
            f"aspect must be 'equal', 'auto', or a positive number, got {aspect!r}"
        )

    dx = abs(np.subtract(*ax.get_xlim()))
    dy = abs(np.subtract(*ax.get_ylim()))
    dz = abs(np.subtract(*ax.get_zlim()))
    ranges = np.maximum([dx, dy, dz], 1e-9)
    ranges[2] *= vertical_exaggeration
    ax.set_box_aspect(tuple(ranges), zoom=zoom)


def _stations_to_local_km(
    dataset: DataSet,
    fault: Fault,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dataset station lat/lon to local Cartesian (East km, North km).

    Uses the fault's reference point as the local origin.
    """
    enu = fault.frame.to_enu(
        lon=dataset.lon,
        lat=dataset.lat,
        alt=np.full(dataset.n_stations, fault.frame.origin_alt),
    )
    return enu[:, 0] * 1e-3, enu[:, 1] * 1e-3


def _get_patch_vertices_local(fault: Fault) -> list[np.ndarray]:
    """Compute 2-D patch vertices in local Cartesian (East km, North km).

    Args:
        fault: Fault geometry.

    Returns:
        List of arrays, each (n_corners, 2). Rectangular patches have
        4 corners; triangular patches have 3.
    """
    if fault.engine == "okada":
        assert fault._length is not None and fault._width is not None
        cos_dip = np.cos(np.radians(fault.dip))
        sin_str = np.sin(np.radians(fault.strike))
        cos_str = np.cos(np.radians(fault.strike))

        half_L = fault._length / 2
        half_W = fault._width / 2

        # Corner offsets in ENU (meters) relative to patch center
        # Order: top-left, top-right, bottom-right, bottom-left
        # "top" = updip (shallower)
        e_off = np.column_stack(
            [
                -half_L * sin_str + half_W * cos_dip * cos_str,
                +half_L * sin_str + half_W * cos_dip * cos_str,
                +half_L * sin_str - half_W * cos_dip * cos_str,
                -half_L * sin_str - half_W * cos_dip * cos_str,
            ]
        )
        n_off = np.column_stack(
            [
                -half_L * cos_str - half_W * cos_dip * sin_str,
                +half_L * cos_str - half_W * cos_dip * sin_str,
                +half_L * cos_str + half_W * cos_dip * sin_str,
                -half_L * cos_str + half_W * cos_dip * sin_str,
            ]
        )

        centers = fault.centers_local
        ce, cn = centers[:, 0], centers[:, 1]

        verts = []
        for i in range(fault.n_patches):
            corners = np.column_stack(
                [
                    (ce[i] + e_off[i]) * 1e-3,
                    (cn[i] + n_off[i]) * 1e-3,
                ]
            )
            verts.append(corners)
        return verts

    # Triangular: _vertices is (N, 3, 3) in local ENU meters [e, n, u]
    tri_verts = fault._vertices
    assert tri_verts is not None
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
    if fault.engine == "okada":
        assert fault._length is not None and fault._width is not None
        sin_dip = np.sin(np.radians(fault.dip))
        cos_dip = np.cos(np.radians(fault.dip))
        sin_str = np.sin(np.radians(fault.strike))
        cos_str = np.cos(np.radians(fault.strike))

        half_L = fault._length / 2
        half_W = fault._width / 2

        e_off = np.column_stack(
            [
                -half_L * sin_str + half_W * cos_dip * cos_str,
                +half_L * sin_str + half_W * cos_dip * cos_str,
                +half_L * sin_str - half_W * cos_dip * cos_str,
                -half_L * sin_str - half_W * cos_dip * cos_str,
            ]
        )
        n_off = np.column_stack(
            [
                -half_L * cos_str - half_W * cos_dip * sin_str,
                +half_L * cos_str - half_W * cos_dip * sin_str,
                +half_L * cos_str + half_W * cos_dip * sin_str,
                -half_L * cos_str + half_W * cos_dip * sin_str,
            ]
        )
        # Depth offsets (positive = deeper)
        d_off = np.column_stack(
            [
                -half_W * sin_dip,
                -half_W * sin_dip,
                +half_W * sin_dip,
                +half_W * sin_dip,
            ]
        )

        centers = fault.centers_local
        ce, cn = centers[:, 0], centers[:, 1]

        verts = []
        for i in range(fault.n_patches):
            corners = np.column_stack(
                [
                    (ce[i] + e_off[i]) * 1e-3,
                    (cn[i] + n_off[i]) * 1e-3,
                    (fault._depth[i] + d_off[i]) * 1e-3,
                ]
            )
            verts.append(corners)
        return verts

    # Triangular
    tri_verts = fault._vertices  # (N, 3, 3) [e, n, u]
    assert tri_verts is not None
    verts = []
    for i in range(fault.n_patches):
        v = tri_verts[i] * 1e-3  # (3, 3) in km
        # Convert up to depth: depth_km = -up_km
        corners = np.column_stack([v[:, 0], v[:, 1], -v[:, 2]])
        verts.append(corners)
    return verts


def _get_patch_vertices_fault(fault: Fault) -> list[np.ndarray]:
    """Compute 2-D patch vertices in fault coordinates (along-strike, along-dip).

    The along-strike axis runs along the fault's average strike direction; the
    along-dip axis measures distance *down the fault plane* from the up-dip
    (shallowest) edge. Both axes are shifted so the shallowest, first
    along-strike corner sits at the origin. Units are kilometers.

    This projection is exact for a planar (uniform strike/dip) fault and a
    reasonable approximation for gently varying meshes.

    Args:
        fault: Fault geometry (rectangular or triangular).

    Returns:
        List of arrays, each (n_corners, 2) with columns
        ``(along_strike_km, along_dip_km)``.
    """
    verts_3d = _get_patch_vertices_3d(fault)  # (east_km, north_km, depth_km down)
    strike_rad = np.radians(float(np.mean(fault.strike)))
    dip_rad = np.radians(float(np.mean(fault.dip)))
    s_hat = np.array([np.sin(strike_rad), np.cos(strike_rad)])
    sin_dip = max(np.sin(dip_rad), 1e-6)

    all_v = np.vstack(verts_3d)
    s0 = float((all_v[:, :2] @ s_hat).min())
    d0 = float((all_v[:, 2] / sin_dip).min())

    return [
        np.column_stack([v[:, :2] @ s_hat - s0, v[:, 2] / sin_dip - d0])
        for v in verts_3d
    ]


def _draw_updip_edge(
    ax: matplotlib.axes.Axes,
    fault: Fault,
    coords: str,
    line_kwargs: dict | None = None,
) -> None:
    """Draw a black line along the fault's up-dip (shallowest) edge.

    In ``'fault'`` coordinates the up-dip edge is the ``along_dip = 0`` line
    spanning the along-strike extent; in ``'geographic'`` coordinates it is the
    surface trace returned by :func:`_get_surface_trace`.
    """
    lkw: dict[str, Any] = {"color": "black", "linewidth": 1.5, "zorder": 6}
    if line_kwargs:
        lkw.update(line_kwargs)

    if coords == "fault":
        all_v = np.vstack(_get_patch_vertices_fault(fault))
        s_min, s_max = float(all_v[:, 0].min()), float(all_v[:, 0].max())
        ax.plot([s_min, s_max], [0.0, 0.0], **lkw)
        return

    trace = _get_surface_trace(fault)
    if trace is not None:
        ax.plot(trace[:, 0], trace[:, 1], **lkw)


def _get_slip_component(
    slip: np.ndarray,
    n_patches: int,
    component: str,
) -> np.ndarray:
    """Extract a scalar per patch from a blocked slip vector.

    If *slip* has length N (single-component vector), it is returned as-is.
    If *slip* has length 2*N (both components), *component* selects which to
    extract.

    Args:
        slip: A single-component vector of length N, or a blocked
            ``[ss_0..ss_N, ds_0..ds_N]`` vector of length 2*N.
        n_patches: Number of fault patches (N).
        component: One of ``'strike'``, ``'dip'``, ``'magnitude'``,
            or ``'magnitude'``.

    Returns:
        Array of shape (N,).

    Raises:
        ValueError: If *component* is invalid or *slip* has wrong length.
    """
    slip_array = np.asarray(slip)
    if slip_array.shape[0] == n_patches:
        return slip_array
    if slip_array.shape[0] != 2 * n_patches:
        raise ValueError(
            f"slip length {slip_array.shape[0]} does not match "
            f"n_patches = {n_patches} or 2 * n_patches = {2 * n_patches}"
        )
    ss = slip_array[:n_patches]
    ds = slip_array[n_patches:]
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
    coords: str = "geographic",
    updip_edge: bool = False,
    updip_edge_kwargs: dict | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot a scalar value on each fault patch as colored polygons.

    This is the shared implementation behind ``slip()``, ``resolution()``,
    ``uncertainty()``, and ``patches()``. See :func:`slip` for a description of
    the ``coords`` and ``updip_edge`` arguments.
    """
    from matplotlib.collections import PolyCollection

    if coords not in ("fault", "geographic"):
        raise ValueError(f"coords must be 'fault' or 'geographic', got {coords!r}")

    ax = _ensure_axes(ax)
    if coords == "fault":
        verts = _get_patch_vertices_fault(fault)
    else:
        verts = _get_patch_vertices_local(fault)

    defaults: dict[str, Any] = {"edgecolor": "face", "linewidth": 0.5}
    defaults.update(kwargs)

    pc = PolyCollection(verts, **defaults)
    pc.set_array(np.asarray(values).ravel())
    pc.set_cmap(cmap)
    if vmin is not None or vmax is not None:
        pc.set_clim(vmin, vmax)

    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    if coords == "fault":
        ax.set_xlabel("Along-strike (km)")
        ax.set_ylabel("Along-dip (km)")
        # Depth increases down-dip; put the up-dip (shallow) edge at the top.
        y0, y1 = ax.get_ylim()
        ax.set_ylim(max(y0, y1), min(y0, y1))
    else:
        ax.set_xlabel("East (km)")
        ax.set_ylabel("North (km)")

    if updip_edge:
        _draw_updip_edge(ax, fault, coords, updip_edge_kwargs)

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
            row_depths = np.array(
                [np.mean(fault._depth[j * nL : (j + 1) * nL]) for j in range(nW)]
            )
            shallow_row = np.argmin(row_depths)
            shallow_indices = list(range(shallow_row * nL, (shallow_row + 1) * nL))
        else:
            # Unstructured: pick patches in the shallowest depth quartile
            depth_threshold = np.percentile(fault._depth, 25)
            shallow_indices = np.where(fault._depth <= depth_threshold)[0].tolist()

        # Collect updip edge endpoints (corners 0 and 1)
        edge_points = []
        for idx in shallow_indices:
            edge_points.append(verts_2d[idx][0])  # top-left
            edge_points.append(verts_2d[idx][1])  # top-right
        if not edge_points:
            return None
        edge_arr = np.array(edge_points)

        # Remove near-duplicates, then sort along the trace.
        # Use PCA-like approach: project onto the dominant direction.
        _, unique_idx = np.unique(np.round(edge_arr, 5), axis=0, return_index=True)
        pts = edge_arr[np.sort(unique_idx)]

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
        threshold = (
            min_depth + depth_range * 0.1 if depth_range > 0 else min_depth + 1.0
        )

        top_points = []
        for v2d, v3d in zip(verts_2d, verts_3d):
            for j in range(len(v3d)):
                if v3d[j, 2] <= threshold:
                    top_points.append(v2d[j])
        if not top_points:
            return None
        top_arr = np.array(top_points)
        _, unique_idx = np.unique(np.round(top_arr, 5), axis=0, return_index=True)
        pts = top_arr[np.sort(unique_idx)]
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


def _patch_centroids_km(fault: Fault) -> tuple[np.ndarray, np.ndarray]:
    """Return per-patch centroids in local map-view coordinates (East, North km)."""
    verts = _get_patch_vertices_local(fault)
    cx = np.array([v[:, 0].mean() for v in verts])
    cy = np.array([v[:, 1].mean() for v in verts])
    return cx, cy


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
    exactly (same head shape, line width, etc.). The legend is placed in a
    dedicated band just outside the data so it never overlaps the plotted
    vectors: the axis limit on the chosen side is extended to make room.
    """
    if label is None:
        label = f"{scale_arrow} observed"

    arrow_km = scale_arrow * scale
    n_rows = 2 if pred_color is not None else 1

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    # Reserve a clear band outside the data on the requested vertical side so
    # the reference arrows never overlap the plotted vectors.
    row_h = y_range * 0.07
    pad = row_h * (n_rows + 1.2)
    if "upper" in loc:
        ax.set_ylim(ylim[0], ylim[1] + pad)
        by = ylim[1] + row_h * 0.8  # observed row just above the data
        row_dy = row_h  # subsequent rows stack further up
    else:
        ax.set_ylim(ylim[0] - pad, ylim[1])
        by = ylim[0] - row_h * 0.8  # observed row just below the data
        row_dy = -row_h  # subsequent rows stack further down

    if "right" in loc:
        bx = xlim[1] - x_range * 0.03
        ha = "right"
    else:
        bx = xlim[0] + x_range * 0.03 + arrow_km
        ha = "left"
    text_x = bx
    text_dy = row_h * 0.15

    # Build quiver kwargs that match the data arrows
    qkw: dict[str, Any] = {"angles": "xy", "scale_units": "xy", "scale": 1}
    if quiver_kwargs:
        qkw.update(quiver_kwargs)

    ax.quiver(bx, by, -arrow_km, 0, color=obs_color, clip_on=False, **qkw)
    ax.text(
        text_x, by + text_dy, label, ha=ha, va="bottom", fontsize=8, color=obs_color
    )

    if pred_color is not None:
        by2 = by + row_dy
        pred_label = (
            label.replace("observed", "predicted")
            if "observed" in label
            else f"{scale_arrow} predicted"
        )
        ax.quiver(bx, by2, -arrow_km, 0, color=pred_color, clip_on=False, **qkw)
        ax.text(
            text_x,
            by2 + text_dy,
            pred_label,
            ha=ha,
            va="bottom",
            fontsize=8,
            color=pred_color,
        )
