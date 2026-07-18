"""Model-assessment plots: resolution and uncertainty maps.

Private submodule of :mod:`geodef.plot`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from geodef.plot._shared import (
    _plot_patch_scalar,
)

if TYPE_CHECKING:
    import matplotlib.axes

    from geodef.fault import Fault


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
    coords: str = "geographic",
    updip_edge: bool = False,
    updip_edge_kwargs: dict | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot resolution matrix diagonal on fault patches.

    Args:
        fault: Fault geometry.
        values: Resolution diagonal, shape (N,). If a full (N, N) resolution
            matrix is supplied, its diagonal is used.
        ax: Axes to plot on. Creates a new figure if ``None``.
        cmap: Matplotlib colormap name.
        vmin: Minimum color limit.
        vmax: Maximum color limit.
        colorbar: Whether to add a colorbar.
        colorbar_label: Colorbar label.
        colorbar_kwargs: Extra kwargs passed to ``fig.colorbar()``.
        title: Axes title.
        coords: Coordinate frame: ``'geographic'`` (default) or ``'fault'``.
            See :func:`slip`.
        updip_edge: Whether to draw a black line along the up-dip edge.
        updip_edge_kwargs: Extra kwargs for the up-dip edge line.
        **kwargs: Passed to ``PolyCollection``.

    Returns:
        The axes used for plotting.
    """
    values = np.asarray(values)
    if values.ndim == 2 and values.shape == (fault.n_patches, fault.n_patches):
        values = np.diag(values)
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
    coords: str = "geographic",
    updip_edge: bool = False,
    updip_edge_kwargs: dict | None = None,
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
        coords: Coordinate frame: ``'geographic'`` (default) or ``'fault'``.
            See :func:`slip`.
        updip_edge: Whether to draw a black line along the up-dip edge.
        updip_edge_kwargs: Extra kwargs for the up-dip edge line.
        **kwargs: Passed to ``PolyCollection``.

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
