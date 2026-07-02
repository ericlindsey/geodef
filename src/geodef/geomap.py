"""Optional geographic (Cartopy) map plotting.

Most ``geodef.plot`` functions work in a local Cartesian frame (East/North km).
This module provides the complementary *geographic* view: a Cartopy map with
coastlines, borders, and optional topography onto which faults and GNSS
vectors can be overlaid in longitude/latitude.

Cartopy is an optional dependency; install it with ``pip install geodef[maps]``.
Every function raises a clear :class:`ImportError` if it is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from geodef import transforms

if TYPE_CHECKING:
    from geodef.data import GNSS
    from geodef.fault import Fault


def _require_cartopy() -> tuple[Any, Any]:
    """Import and return ``(cartopy.crs, cartopy.feature)`` or raise."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:  # pragma: no cover - exercised only without cartopy
        raise ImportError(
            "Geographic plotting requires Cartopy. "
            "Install it with 'pip install geodef[maps]'."
        ) from exc
    return ccrs, cfeature


def basemap(
    extent: tuple[float, float, float, float] | None = None,
    *,
    ax: Any = None,
    coastlines: bool = True,
    borders: bool = False,
    land: bool = False,
    ocean: bool = False,
    stock_img: bool = False,
    resolution: str = "50m",
    gridlines: bool = True,
    figsize: tuple[float, float] = (8, 8),
) -> Any:
    """Create a Cartopy map axes with the requested background features.

    The axes use a PlateCarree projection, so subsequent lon/lat plotting can
    use ``transform=cartopy.crs.PlateCarree()`` (the overlay helpers in this
    module do this for you).

    Args:
        extent: ``(lon_min, lon_max, lat_min, lat_max)`` in degrees. If None,
            Cartopy auto-scales to the plotted data.
        ax: Existing Cartopy ``GeoAxes`` to draw on. A new figure is created
            if None.
        coastlines: Draw coastlines.
        borders: Draw national borders.
        land: Fill land polygons.
        ocean: Fill ocean polygons.
        stock_img: Add Cartopy's built-in low-resolution shaded relief
            (no network download).
        resolution: Natural Earth feature resolution (``'10m'``, ``'50m'``,
            ``'110m'``).
        gridlines: Draw labeled lon/lat gridlines.
        figsize: Figure size when creating a new figure.

    Returns:
        A Cartopy ``GeoAxes``.
    """
    ccrs, cfeature = _require_cartopy()
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    if stock_img:
        ax.stock_img()
    if land:
        ax.add_feature(cfeature.LAND.with_scale(resolution))
    if ocean:
        ax.add_feature(cfeature.OCEAN.with_scale(resolution))
    if coastlines:
        ax.coastlines(resolution=resolution)
    if borders:
        ax.add_feature(cfeature.BORDERS.with_scale(resolution), linewidth=0.5)
    if gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

    return ax


def _patch_outlines_lonlat(fault: Fault) -> np.ndarray:
    """Return closed per-patch outlines as ``(N, k+1, 2)`` in ``[lon, lat]``."""
    if fault.engine == "okada":
        return fault.patch_outlines
    # Triangular: convert ENU vertices to geographic and close each triangle.
    verts = fault._vertices  # (N, 3, 3) as [e, n, u]
    assert verts is not None
    n_tri = verts.shape[0]
    lon, lat, _ = transforms.enu2geod(
        verts[:, :, 0].ravel(),
        verts[:, :, 1].ravel(),
        verts[:, :, 2].ravel(),
        fault._ref_lat,
        fault._ref_lon,
        0.0,
    )
    lonlat = np.stack([lon, lat], axis=1).reshape(n_tri, 3, 2)
    return np.concatenate([lonlat, lonlat[:, :1, :]], axis=1)


def add_fault(
    ax: Any,
    fault: Fault,
    *,
    edgecolor: str = "red",
    facecolor: str = "none",
    linewidth: float = 0.8,
    **kwargs: Any,
) -> Any:
    """Overlay a fault's patch outlines on a geographic axes.

    Args:
        ax: A Cartopy ``GeoAxes`` (e.g. from :func:`basemap`).
        fault: Fault geometry (rectangular or triangular).
        edgecolor: Outline color.
        facecolor: Fill color (``'none'`` for outlines only).
        linewidth: Outline width.
        **kwargs: Extra kwargs passed to ``PolyCollection``.

    Returns:
        The added ``PolyCollection``.
    """
    ccrs, _ = _require_cartopy()
    from matplotlib.collections import PolyCollection

    outlines = _patch_outlines_lonlat(fault)
    pc = PolyCollection(
        list(outlines),
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        transform=ccrs.PlateCarree(),
        **kwargs,
    )
    ax.add_collection(pc)
    return pc


def add_vectors(
    ax: Any,
    dataset: GNSS,
    *,
    scale: float = 1.0,
    color: str = "black",
    **kwargs: Any,
) -> Any:
    """Overlay horizontal GNSS velocity vectors on a geographic axes.

    Args:
        ax: A Cartopy ``GeoAxes`` (e.g. from :func:`basemap`).
        dataset: A ``GNSS`` dataset with East/North components.
        scale: Multiplier applied to the (E, N) components before plotting.
        color: Arrow color.
        **kwargs: Extra kwargs passed to ``quiver``.

    Returns:
        The ``Quiver`` artist.
    """
    ccrs, _ = _require_cartopy()
    return ax.quiver(
        dataset.lon,
        dataset.lat,
        dataset._ve * scale,
        dataset._vn * scale,
        color=color,
        transform=ccrs.PlateCarree(),
        angles="xy",
        **kwargs,
    )
