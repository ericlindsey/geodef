"""Fault-geometry and slip plots.

Private submodule of :mod:`geodef.plot`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from geodef.plot._shared import (
    _apply_3d_aspect,
    _ensure_axes,
    _ensure_axes_3d,
    _get_patch_vertices_3d,
    _get_slip_component,
    _patch_centroids_km,
    _plot_patch_scalar,
    _stations_to_local_km,
)

if TYPE_CHECKING:
    import matplotlib.axes
    from mpl_toolkits.mplot3d import Axes3D

    from geodef.data import DataSet
    from geodef.fault import Fault


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
    coords: str = "geographic",
    updip_edge: bool = False,
    updip_edge_kwargs: dict | None = None,
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
        coords: Coordinate frame for the axes: ``'geographic'`` (default,
            East/North km) or ``'fault'`` (along-strike/along-dip km). See
            :func:`slip`.
        updip_edge: Whether to draw a black line along the up-dip edge.
        updip_edge_kwargs: Extra kwargs for the up-dip edge line.
        **kwargs: Passed to ``PolyCollection`` (e.g. ``edgecolor``,
            ``linewidth``).

    Returns:
        The axes used for plotting.
    """
    return _plot_patch_scalar(
        fault,
        values,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
        colorbar_kwargs=colorbar_kwargs,
        title=title,
        coords=coords,
        updip_edge=updip_edge,
        updip_edge_kwargs=updip_edge_kwargs,
        **kwargs,
    )


def slip(
    fault: Fault,
    slip_vector: np.ndarray,
    *,
    ax: matplotlib.axes.Axes | None = None,
    components: str = "magnitude",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    colorbar_kwargs: dict | None = None,
    title: str | None = None,
    coords: str = "fault",
    updip_edge: bool = True,
    updip_edge_kwargs: dict | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot fault slip distribution as colored patches.

    Args:
        fault: Fault geometry (rectangular or triangular).
        slip_vector: A length-N single component or a length-2*N blocked
            ``[ss_0..ss_N, ds_0..ds_N]`` vector.
        ax: Axes to plot on. Creates a new figure if ``None``.
        components: Which component to display when *slip_vector* has length
            2*N. One of ``'strike'``, ``'dip'``, or ``'magnitude'``
            (default). Ignored for single-component vectors.
        cmap: Matplotlib colormap name.
        vmin: Minimum color limit.
        vmax: Maximum color limit.
        colorbar: Whether to add a colorbar.
        colorbar_label: Colorbar label. Auto-generated if ``None``.
        colorbar_kwargs: Extra kwargs passed to ``fig.colorbar()``.
        title: Axes title.
        coords: Coordinate frame for the axes. ``'fault'`` (default) draws the
            slip in along-strike / along-dip kilometers with the up-dip edge at
            the top — the natural frame for reading a slip distribution.
            ``'geographic'`` uses local East/North kilometers, which is needed
            when overlaying the slip on a map alongside :func:`vectors`,
            :func:`insar`, or station locations.
        updip_edge: Whether to draw a black line along the up-dip (shallowest)
            fault edge. Defaults to ``True``.
        updip_edge_kwargs: Extra kwargs for the up-dip edge line (e.g.
            ``color``, ``linewidth``).
        **kwargs: Passed to ``PolyCollection`` (e.g. ``edgecolor``,
            ``linewidth``).

    Returns:
        The axes used for plotting.
    """
    values = _get_slip_component(slip_vector, fault.n_patches, components)
    if colorbar_label is None:
        labels = {
            "strike": "Strike-slip (m)",
            "dip": "Dip-slip (m)",
            "magnitude": "Slip magnitude (m)",
        }
        colorbar_label = labels.get(components, "Slip (m)")
    return _plot_patch_scalar(
        fault,
        values,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
        colorbar_kwargs=colorbar_kwargs,
        title=title,
        coords=coords,
        updip_edge=updip_edge,
        updip_edge_kwargs=updip_edge_kwargs,
        **kwargs,
    )


def slip_interpolated(
    fault: Fault,
    slip_vector: np.ndarray,
    *,
    ax: matplotlib.axes.Axes | None = None,
    components: str = "magnitude",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    levels: int = 20,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    colorbar_kwargs: dict | None = None,
    title: str | None = None,
) -> matplotlib.axes.Axes:
    """Plot a smoothly interpolated slip field in map view.

    Unlike :func:`slip`, which draws discrete colored patches, this renders a
    continuous field: a ``pcolormesh`` with Gouraud shading over the structured
    grid for rectangular faults, or a ``tricontourf`` over the patch centroids
    for triangular (or unstructured) faults.

    Args:
        fault: Fault geometry (rectangular or triangular).
        slip_vector: Slip vector (see :func:`slip`).
        ax: Axes to plot on. Creates a new figure if ``None``.
        components: Component to display for a 2*N vector: ``'strike'``,
            ``'dip'``, or ``'magnitude'`` (default).
        cmap: Matplotlib colormap name.
        vmin: Minimum color limit.
        vmax: Maximum color limit.
        levels: Number of filled contour levels (``tricontourf`` path only).
        colorbar: Whether to add a colorbar.
        colorbar_label: Colorbar label. Auto-generated if ``None``.
        colorbar_kwargs: Extra kwargs passed to ``fig.colorbar()``.
        title: Axes title.

    Returns:
        The axes used for plotting.
    """
    values = _get_slip_component(slip_vector, fault.n_patches, components)
    if colorbar_label is None:
        labels = {
            "strike": "Strike-slip (m)",
            "dip": "Dip-slip (m)",
            "magnitude": "Slip magnitude (m)",
            "rake_parallel": "Rake-parallel slip (m)",
            "rake_perpendicular": "Rake-perpendicular slip (m)",
        }
        colorbar_label = labels.get(components, "Slip (m)")

    ax = _ensure_axes(ax)
    cx, cy = _patch_centroids_km(fault)

    mesh: Any
    if fault.engine == "okada" and fault.grid_shape is not None:
        n_length, n_width = fault.grid_shape
        shape = (n_width, n_length)
        mesh = ax.pcolormesh(
            cx.reshape(shape),
            cy.reshape(shape),
            values.reshape(shape),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="gouraud",
        )
    else:
        mesh = ax.tricontourf(
            cx, cy, values, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax
        )

    ax.set_aspect("equal")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    if title is not None:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=colorbar_label, **(colorbar_kwargs or {}))
    return ax


def fault3d(
    fault: Fault,
    *,
    ax: Axes3D | None = None,
    color_by: str | np.ndarray | None = "depth",
    cmap: str = "viridis",
    show_edges: bool = True,
    station_locations: DataSet | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    view: tuple[float, float] | None = None,
    aspect: str | float = "equal",
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
        aspect: Data aspect ratio. ``'equal'`` (default) scales all three
            axes in proportion to their data ranges so geometry is not
            distorted; ``'auto'`` uses matplotlib's default cubic box; a
            positive number applies that vertical exaggeration to the
            depth axis relative to equal horizontal scaling (useful for
            shallow faults whose depth extent is small).
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

    defaults: dict[str, Any] = {
        "edgecolor": "gray" if show_edges else "none",
        "linewidth": 0.5 if show_edges else 0,
        "alpha": 0.8,
    }
    defaults.update(kwargs)

    pc = Poly3DCollection(verts_3d, **defaults)

    if face_values is not None:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        norm = mcolors.Normalize(vmin=np.min(face_values), vmax=np.max(face_values))
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        pc.set_facecolor(mapper.to_rgba(face_values))

        if colorbar:
            # ``pad`` keeps the colorbar clear of the depth axis labels,
            # which sit on the right of the box at typical view angles.
            ax.figure.colorbar(mapper, ax=ax, label=colorbar_label, shrink=0.6, pad=0.1)

    ax.add_collection3d(pc)

    # Station locations — plot BEFORE setting limits so they're included.
    # Use computed_zorder=False to force manual zorder, which is the best
    # available workaround for matplotlib's 3D rendering order issue.
    if station_locations is not None:
        ax.computed_zorder = False
        pc.set_zorder(1)
        sx, sy = _stations_to_local_km(station_locations, fault)
        ax.scatter(
            sx,
            sy,
            np.zeros_like(sx),
            c="red",
            s=30,
            marker="^",
            zorder=10,
            depthshade=False,
            label="Stations",
        )

    # Set axis limits from vertices
    all_verts = np.vstack(verts_3d)
    pad = 0.05
    for dim, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        lo, hi = all_verts[:, dim].min(), all_verts[:, dim].max()
        margin = (hi - lo) * pad or 1.0
        setter(lo - margin, hi + margin)

    # Invert z so depth increases downward
    if ax.get_zlim()[0] < ax.get_zlim()[1]:
        ax.invert_zaxis()

    _apply_3d_aspect(ax, aspect)

    # Thin in depth: cap depth ticks and pad the label so numbers neither
    # crowd each other nor collide with the North-axis tick labels.
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.labelpad = 12

    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_zlabel("Depth (km)")

    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])

    if title is not None:
        ax.set_title(title)

    return ax
