"""Tests for geodef.geomap geographic (Cartopy) plotting.

Cartopy is an optional dependency; these tests are skipped when it is not
installed, but the module itself must always import.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from geodef import geomap
from geodef.data import GNSS
from geodef.fault import Fault

cartopy = pytest.importorskip("cartopy")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def rect_fault():
    return Fault.planar(
        lat=0.0,
        lon=100.0,
        depth=15e3,
        strike=320.0,
        dip=15.0,
        length=80e3,
        width=40e3,
        n_length=4,
        n_width=3,
    )


@pytest.fixture
def tri_fault():
    nodes = np.array([[0.0, 0, 0], [1e4, 0, 0], [0, 1e4, -5e3], [1e4, 1e4, -5e3]])
    tris = np.array([[0, 1, 2], [1, 3, 2]])
    return Fault.from_triangles(nodes, ref_lat=0.0, ref_lon=100.0, triangles=tris)


class TestBasemap:
    def test_returns_geoaxes(self):
        ax = geomap.basemap(extent=(99, 101, -1, 1))
        assert hasattr(ax, "coastlines")

    def test_all_features(self):
        ax = geomap.basemap(
            extent=(99, 101, -1, 1),
            coastlines=True,
            borders=True,
            land=True,
            ocean=True,
            gridlines=True,
        )
        assert ax is not None

    def test_existing_axes(self):
        import cartopy.crs as ccrs

        fig = plt.figure()
        ax_in = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax_out = geomap.basemap(ax=ax_in, coastlines=False, gridlines=False)
        assert ax_out is ax_in


class TestAddFault:
    def test_okada(self, rect_fault):
        ax = geomap.basemap(coastlines=False, gridlines=False)
        pc = geomap.add_fault(ax, rect_fault)
        assert pc in ax.collections

    def test_tri(self, tri_fault):
        ax = geomap.basemap(coastlines=False, gridlines=False)
        pc = geomap.add_fault(ax, tri_fault, edgecolor="blue")
        assert pc in ax.collections


class TestAddVectors:
    def test_quiver(self, rect_fault):
        ax = geomap.basemap(coastlines=False, gridlines=False)
        n = 5
        g = GNSS(
            lon=np.linspace(99.5, 100.5, n),
            lat=np.zeros(n),
            ve=np.ones(n) * 0.01,
            vn=np.ones(n) * 0.02,
            vu=None,
            se=np.ones(n),
            sn=np.ones(n),
            su=None,
        )
        q = geomap.add_vectors(ax, g, scale=1000)
        assert q is not None
