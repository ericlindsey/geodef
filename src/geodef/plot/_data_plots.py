"""Observation-data plots: GNSS vectors, InSAR scatter, map view.

Private submodule of :mod:`geodef.plot`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from geodef.plot._shared import (
    _add_scale_arrow_legend,
    _ensure_axes,
    _get_patch_vertices_local,
    _get_slip_component,
    _get_surface_trace,
    _stations_to_local_km,
)

if TYPE_CHECKING:
    import matplotlib.axes

    from geodef.data import GNSS, DataSet, InSAR
    from geodef.fault import Fault


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
    vertical_size: float = 40.0,
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
            ``scale`` affects only the horizontal arrows; the vertical
            dots are drawn at a constant size (see ``vertical_size``).
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
            component when vertical dots are drawn (``components='vertical'``
            or ``'both'``).
        vertical_colorbar_label: Label for the vertical colorbar.
        vertical_size: Constant marker area (points^2) for the vertical
            dots. The vertical component is encoded by color (symmetric
            ``RdBu_r``), not marker size.
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
        raise ValueError("Vertical component requested but dataset is horizontal-only.")

    # Arrows sit above the vertical dots (zorder 5) so that large dots can't
    # hide the displacement vectors in ``components='both'`` mode.
    qkw: dict[str, Any] = {"angles": "xy", "scale_units": "xy", "scale": 1, "zorder": 5}
    if quiver_kwargs:
        qkw.update(quiver_kwargs)

    if components in ("horizontal", "both"):
        ve = dataset._ve * scale
        vn = dataset._vn * scale
        ax.quiver(x_km, y_km, ve, vn, color=obs_color, label="_nolegend_", **qkw)

        if predicted is not None:
            if has_vert:
                pe = predicted[0::3] * scale
                pn = predicted[1::3] * scale
            else:
                pe = predicted[0::2] * scale
                pn = predicted[1::2] * scale
            ax.quiver(x_km, y_km, pe, pn, color=pred_color, label="_nolegend_", **qkw)

        if ellipses:
            ekw: dict[str, Any] = {
                "facecolor": "none",
                "edgecolor": obs_color,
                "linewidth": 0.5,
                "alpha": 0.4,
            }
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
        vu = dataset._vu  # raw values — encode magnitude by color only
        assert vu is not None
        # Vertical dots are a constant size; the vertical component is read
        # from color, not marker area (value-scaled areas produced dots that
        # were far too large).
        absmax = float(np.max(np.abs(vu))) if vu.size else 0.0
        clim = absmax if absmax > 0 else None

        sc = ax.scatter(
            x_km,
            y_km,
            c=vu,
            s=vertical_size,
            cmap="RdBu_r",
            vmin=-clim if clim is not None else None,
            vmax=clim if clim is not None else None,
            edgecolors="k",
            linewidths=0.5,
            zorder=2,
            label="Vertical (obs)" if components == "vertical" else None,
        )
        if vertical_colorbar and components in ("vertical", "both"):
            ax.figure.colorbar(sc, ax=ax, label=vertical_colorbar_label)

        if predicted is not None and has_vert:
            ax.scatter(
                x_km,
                y_km,
                s=vertical_size,
                edgecolors=pred_color,
                linewidths=1.0,
                facecolors="none",
                marker="o",
                zorder=3,
            )

    ax.set_aspect("equal")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    if title is not None:
        ax.set_title(title)

    # Finalize data limits before placing the scale arrow so it is anchored to
    # the true corner of the plotted data, not the pre-autoscale extent.
    ax.autoscale_view()

    if legend:
        if scale_arrow is not None:
            _add_scale_arrow_legend(
                ax,
                scale_arrow,
                scale,
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
                Line2D([0], [0], color=obs_color, linewidth=2, label="Observed"),
            ]
            if predicted is not None:
                handles.append(
                    Line2D([0], [0], color=pred_color, linewidth=2, label="Predicted"),
                )
            ax.legend(handles=handles)

    return ax


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
    skw: dict[str, Any] = {"s": 8, "cmap": cmap}
    if scatter_kwargs:
        skw.update(scatter_kwargs)
    # Don't pass cmap twice
    skw.setdefault("cmap", cmap)

    def _auto_clim(data: np.ndarray):
        if vmin is not None and vmax is not None:
            return vmin, vmax
        absmax = np.max(np.abs(data))
        return (
            vmin if vmin is not None else -absmax,
            vmax if vmax is not None else absmax,
        )

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
        _scatter_panel(axes[1], predicted, "Predicted", show_ylabel=False)
        sc2 = _scatter_panel(axes[2], residual, "Residual", show_ylabel=False)
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
        assert predicted is not None  # guaranteed by the check above
        data = predicted
        panel_title = title or "Predicted"
    elif layout == "residual":
        assert predicted is not None  # guaranteed by the check above
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


def map_view(
    fault: Fault,
    *,
    datasets: DataSet | list[DataSet] | None = None,
    values: np.ndarray | None = None,
    slip_vector: np.ndarray | None = None,
    components: str = "magnitude",
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
    (a length-*n_patches* array) or ``slip_vector`` (either one amplitude
    per patch, or a blocked ``[ss | ds]`` vector decomposed via
    ``components``).
    When neither is provided, patches are drawn with a uniform fill.

    Args:
        fault: Fault geometry.
        datasets: One or more datasets whose station locations are
            overlaid on the map.
        values: Per-patch scalar array (length *n_patches*) to color
            the patches by. Mutually exclusive with ``slip_vector``.
        slip_vector: Length *n_patches* vector or blocked ``[ss | ds]`` length
            *2 × n_patches*. Decomposed via ``components`` when blocked.
        components: Which slip component to extract when using
            ``slip_vector``. One of ``'magnitude'``, ``'strike'``,
            ``'dip'`` (default ``'magnitude'``). Ignored for
            single-amplitude vectors.
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
        color_values = _get_slip_component(slip_vector, fault.n_patches, components)

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
            tkw: dict[str, Any] = {"color": "red", "linewidth": 2, "zorder": 5}
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
            ax.scatter(sx, sy, marker=mk, color=cl, s=30, zorder=10, label=label)

    ax.set_aspect("equal")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.autoscale_view()
    if title is not None:
        ax.set_title(title)

    return ax
