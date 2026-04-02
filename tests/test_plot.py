"""Tests for geodef.plot visualization module."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pytest

import geodef
from geodef.data import GNSS, InSAR, Vertical
from geodef.fault import Fault
from geodef.invert import InversionResult


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def rect_fault():
    """Simple 2x3 rectangular planar fault."""
    return Fault.planar(
        lat=0.0, lon=0.0, depth=10_000.0,
        strike=0.0, dip=45.0,
        length=30_000.0, width=20_000.0,
        n_length=3, n_width=2,
    )


@pytest.fixture
def tri_fault():
    """Simple triangular fault with 2 patches."""
    n = 2
    lat = np.array([0.0, 0.01])
    lon = np.array([0.0, 0.01])
    depth = np.array([5000.0, 8000.0])
    strike = np.array([0.0, 0.0])
    dip = np.array([45.0, 45.0])

    vertices = np.array([
        [[0.0, 0.0, -5000.0],
         [5000.0, 0.0, -5000.0],
         [2500.0, 5000.0, -8000.0]],
        [[5000.0, 0.0, -5000.0],
         [10000.0, 0.0, -5000.0],
         [7500.0, 5000.0, -8000.0]],
    ])
    return Fault(
        lat, lon, depth, strike, dip,
        length=None, width=None,
        vertices=vertices, engine="tri",
    )


@pytest.fixture
def slip_magnitude(rect_fault):
    """Slip vector (strike + dip) for a rectangular fault."""
    n = rect_fault.n_patches
    return np.concatenate([np.ones(n) * 0.5, np.ones(n) * 1.0])


@pytest.fixture
def slip_tri(tri_fault):
    """Slip vector for triangular fault."""
    n = tri_fault.n_patches
    return np.concatenate([np.ones(n) * 0.3, np.ones(n) * 0.7])


@pytest.fixture
def gnss_3comp():
    """3-component GNSS dataset with 5 stations."""
    rng = np.random.default_rng(42)
    n = 5
    lat = rng.uniform(-0.1, 0.1, n)
    lon = rng.uniform(-0.1, 0.1, n)
    ve = rng.normal(0, 0.01, n)
    vn = rng.normal(0, 0.01, n)
    vu = rng.normal(0, 0.005, n)
    se = np.full(n, 0.001)
    sn = np.full(n, 0.001)
    su = np.full(n, 0.002)
    return GNSS(lon, lat, ve, vn, vu, se, sn, su)


@pytest.fixture
def gnss_horiz():
    """Horizontal-only GNSS dataset."""
    rng = np.random.default_rng(43)
    n = 5
    lat = rng.uniform(-0.1, 0.1, n)
    lon = rng.uniform(-0.1, 0.1, n)
    ve = rng.normal(0, 0.01, n)
    vn = rng.normal(0, 0.01, n)
    se = np.full(n, 0.001)
    sn = np.full(n, 0.001)
    return GNSS(lon, lat, ve, vn, None, se, sn, None)


@pytest.fixture
def insar_data():
    """InSAR LOS dataset with 20 points."""
    rng = np.random.default_rng(44)
    n = 20
    lat = rng.uniform(-0.1, 0.1, n)
    lon = rng.uniform(-0.1, 0.1, n)
    los = rng.normal(0, 0.05, n)
    sigma = np.full(n, 0.01)
    # Near-vertical look vector (typical ascending geometry)
    look_e = np.full(n, -0.1)
    look_n = np.full(n, 0.08)
    look_u = np.full(n, 0.99)
    return InSAR(lon, lat, los, sigma, look_e, look_n, look_u)


@pytest.fixture
def vertical_data():
    """Vertical dataset with 8 points."""
    rng = np.random.default_rng(45)
    n = 8
    lat = rng.uniform(-0.1, 0.1, n)
    lon = rng.uniform(-0.1, 0.1, n)
    obs = rng.normal(0, 0.02, n)
    sigma = np.full(n, 0.005)
    return Vertical(lon, lat, obs, sigma)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


# ======================================================================
# Internal helpers
# ======================================================================

class TestGetPatchVerticesLocal:
    """Tests for _get_patch_vertices_local."""

    def test_rect_shape(self, rect_fault):
        from geodef.plot import _get_patch_vertices_local
        verts = _get_patch_vertices_local(rect_fault)
        assert len(verts) == rect_fault.n_patches
        assert verts[0].shape == (4, 2)

    def test_tri_shape(self, tri_fault):
        from geodef.plot import _get_patch_vertices_local
        verts = _get_patch_vertices_local(tri_fault)
        assert len(verts) == tri_fault.n_patches
        assert verts[0].shape == (3, 2)

    def test_rect_units_km(self, rect_fault):
        from geodef.plot import _get_patch_vertices_local
        verts = _get_patch_vertices_local(rect_fault)
        all_x = np.concatenate([v[:, 0] for v in verts])
        all_y = np.concatenate([v[:, 1] for v in verts])
        assert np.ptp(all_x) > 1.0
        assert np.ptp(all_x) < 100.0
        assert np.ptp(all_y) > 1.0
        assert np.ptp(all_y) < 100.0

    def test_tri_units_km(self, tri_fault):
        from geodef.plot import _get_patch_vertices_local
        verts = _get_patch_vertices_local(tri_fault)
        all_x = np.concatenate([v[:, 0] for v in verts])
        assert np.ptp(all_x) == pytest.approx(10.0, abs=1.0)


class TestGetSlipComponent:
    """Tests for _get_slip_component."""

    def test_magnitude(self, rect_fault, slip_magnitude):
        from geodef.plot import _get_slip_component
        n = rect_fault.n_patches
        vals = _get_slip_component(slip_magnitude, n, "magnitude")
        assert vals.shape == (n,)
        expected = np.sqrt(0.5**2 + 1.0**2)
        np.testing.assert_allclose(vals, expected)

    def test_strike(self, rect_fault, slip_magnitude):
        from geodef.plot import _get_slip_component
        n = rect_fault.n_patches
        vals = _get_slip_component(slip_magnitude, n, "strike")
        assert vals.shape == (n,)
        np.testing.assert_allclose(vals, 0.5)

    def test_dip(self, rect_fault, slip_magnitude):
        from geodef.plot import _get_slip_component
        n = rect_fault.n_patches
        vals = _get_slip_component(slip_magnitude, n, "dip")
        assert vals.shape == (n,)
        np.testing.assert_allclose(vals, 1.0)

    def test_invalid_component(self, rect_fault, slip_magnitude):
        from geodef.plot import _get_slip_component
        with pytest.raises(ValueError, match="component"):
            _get_slip_component(slip_magnitude, rect_fault.n_patches, "invalid")

    def test_wrong_slip_length(self, rect_fault):
        from geodef.plot import _get_slip_component
        with pytest.raises(ValueError, match="length"):
            _get_slip_component(np.ones(5), rect_fault.n_patches, "magnitude")


class TestStationsToLocal:
    """Tests for _stations_to_local_km."""

    def test_returns_two_arrays(self, rect_fault, gnss_3comp):
        from geodef.plot import _stations_to_local_km
        x, y = _stations_to_local_km(gnss_3comp, rect_fault)
        assert x.shape == (gnss_3comp.n_stations,)
        assert y.shape == (gnss_3comp.n_stations,)

    def test_units_km(self, rect_fault, gnss_3comp):
        from geodef.plot import _stations_to_local_km
        x, y = _stations_to_local_km(gnss_3comp, rect_fault)
        assert np.all(np.abs(x) < 50)
        assert np.all(np.abs(y) < 50)


# ======================================================================
# plot.patches (generic per-patch scalar)
# ======================================================================

class TestPlotPatches:
    """Tests for geodef.plot.patches."""

    def test_returns_axes(self, rect_fault):
        values = rect_fault._depth * 1e-3
        ax = geodef.plot.patches(rect_fault, values)
        assert isinstance(ax, plt.Axes)

    def test_custom_label(self, rect_fault):
        values = rect_fault._depth * 1e-3
        ax = geodef.plot.patches(rect_fault, values,
                                  colorbar_label="Depth (km)")
        assert isinstance(ax, plt.Axes)

    def test_tri_fault(self, tri_fault):
        values = np.array([1.0, 2.0])
        ax = geodef.plot.patches(tri_fault, values, cmap="coolwarm")
        assert isinstance(ax, plt.Axes)

    def test_existing_axes(self, rect_fault):
        fig, ax_in = plt.subplots()
        values = np.ones(rect_fault.n_patches)
        ax_out = geodef.plot.patches(rect_fault, values, ax=ax_in)
        assert ax_out is ax_in

    def test_kwargs_passthrough(self, rect_fault):
        values = np.ones(rect_fault.n_patches)
        ax = geodef.plot.patches(rect_fault, values,
                                  edgecolor="blue", linewidth=2)
        assert isinstance(ax, plt.Axes)


# ======================================================================
# plot.slip
# ======================================================================

class TestPlotSlip:
    """Tests for geodef.plot.slip."""

    def test_returns_axes(self, rect_fault, slip_magnitude):
        ax = geodef.plot.slip(rect_fault, slip_magnitude)
        assert isinstance(ax, plt.Axes)

    def test_rect_fault(self, rect_fault, slip_magnitude):
        ax = geodef.plot.slip(rect_fault, slip_magnitude)
        assert len(ax.collections) >= 1

    def test_tri_fault(self, tri_fault, slip_tri):
        ax = geodef.plot.slip(tri_fault, slip_tri)
        assert isinstance(ax, plt.Axes)
        assert len(ax.collections) >= 1

    def test_component_strike(self, rect_fault, slip_magnitude):
        ax = geodef.plot.slip(rect_fault, slip_magnitude, component="strike")
        assert isinstance(ax, plt.Axes)

    def test_component_dip(self, rect_fault, slip_magnitude):
        ax = geodef.plot.slip(rect_fault, slip_magnitude, component="dip")
        assert isinstance(ax, plt.Axes)

    def test_existing_axes(self, rect_fault, slip_magnitude):
        fig, ax_in = plt.subplots()
        ax_out = geodef.plot.slip(rect_fault, slip_magnitude, ax=ax_in)
        assert ax_out is ax_in

    def test_colorbar_default(self, rect_fault, slip_magnitude):
        ax = geodef.plot.slip(rect_fault, slip_magnitude)
        assert len(ax.figure.get_axes()) >= 2

    def test_colorbar_disabled(self, rect_fault, slip_magnitude):
        ax = geodef.plot.slip(rect_fault, slip_magnitude, colorbar=False)
        assert len(ax.figure.get_axes()) == 1

    def test_colorbar_label(self, rect_fault, slip_magnitude):
        ax = geodef.plot.slip(rect_fault, slip_magnitude, colorbar_label="Test label")
        assert len(ax.figure.get_axes()) >= 2

    def test_custom_cmap(self, rect_fault, slip_magnitude):
        ax = geodef.plot.slip(rect_fault, slip_magnitude, cmap="RdBu_r")
        assert isinstance(ax, plt.Axes)

    def test_vmin_vmax(self, rect_fault, slip_magnitude):
        ax = geodef.plot.slip(rect_fault, slip_magnitude, vmin=0, vmax=2)
        coll = ax.collections[0]
        assert coll.get_clim() == (0, 2)

    def test_kwargs_passthrough(self, rect_fault, slip_magnitude):
        ax = geodef.plot.slip(rect_fault, slip_magnitude,
                              edgecolor="red", linewidth=2.0)
        assert isinstance(ax, plt.Axes)

    def test_title(self, rect_fault, slip_magnitude):
        ax = geodef.plot.slip(rect_fault, slip_magnitude, title="My title")
        assert ax.get_title() == "My title"

    def test_equal_aspect(self, rect_fault, slip_magnitude):
        ax = geodef.plot.slip(rect_fault, slip_magnitude)
        assert ax.get_aspect() in ("equal", 1.0)

    def test_axis_labels(self, rect_fault, slip_magnitude):
        ax = geodef.plot.slip(rect_fault, slip_magnitude)
        assert "km" in ax.get_xlabel().lower() or "east" in ax.get_xlabel().lower()
        assert "km" in ax.get_ylabel().lower() or "north" in ax.get_ylabel().lower()

    def test_colorbar_kwargs(self, rect_fault, slip_magnitude):
        ax = geodef.plot.slip(rect_fault, slip_magnitude,
                              colorbar_kwargs={"orientation": "horizontal"})
        assert isinstance(ax, plt.Axes)


# ======================================================================
# plot.resolution and plot.uncertainty
# ======================================================================

class TestPlotResolution:
    """Tests for geodef.plot.resolution."""

    def test_returns_axes(self, rect_fault):
        values = np.random.rand(rect_fault.n_patches)
        ax = geodef.plot.resolution(rect_fault, values)
        assert isinstance(ax, plt.Axes)

    def test_default_clim(self, rect_fault):
        values = np.random.rand(rect_fault.n_patches)
        ax = geodef.plot.resolution(rect_fault, values)
        coll = ax.collections[0]
        assert coll.get_clim() == (0, 1)

    def test_custom_kwargs(self, rect_fault):
        values = np.random.rand(rect_fault.n_patches)
        ax = geodef.plot.resolution(rect_fault, values,
                                     cmap="plasma", edgecolor="white")
        assert isinstance(ax, plt.Axes)

    def test_tri_fault(self, tri_fault):
        values = np.random.rand(tri_fault.n_patches)
        ax = geodef.plot.resolution(tri_fault, values)
        assert isinstance(ax, plt.Axes)


class TestPlotUncertainty:
    """Tests for geodef.plot.uncertainty."""

    def test_returns_axes(self, rect_fault):
        values = np.random.rand(rect_fault.n_patches) * 0.1
        ax = geodef.plot.uncertainty(rect_fault, values)
        assert isinstance(ax, plt.Axes)

    def test_tri_fault(self, tri_fault):
        values = np.random.rand(tri_fault.n_patches) * 0.1
        ax = geodef.plot.uncertainty(tri_fault, values)
        assert isinstance(ax, plt.Axes)


# ======================================================================
# plot.vectors (6.7c)
# ======================================================================

class TestPlotVectors:
    """Tests for geodef.plot.vectors."""

    def test_returns_axes(self, gnss_3comp, rect_fault):
        ax = geodef.plot.vectors(gnss_3comp, rect_fault)
        assert isinstance(ax, plt.Axes)

    def test_horizontal_only_gnss(self, gnss_horiz, rect_fault):
        ax = geodef.plot.vectors(gnss_horiz, rect_fault)
        assert isinstance(ax, plt.Axes)

    def test_with_predicted(self, gnss_3comp, rect_fault):
        predicted = gnss_3comp.obs * 0.9  # approximate prediction
        ax = geodef.plot.vectors(gnss_3comp, rect_fault, predicted=predicted)
        assert isinstance(ax, plt.Axes)

    def test_existing_axes(self, gnss_3comp, rect_fault):
        fig, ax_in = plt.subplots()
        ax_out = geodef.plot.vectors(gnss_3comp, rect_fault, ax=ax_in)
        assert ax_out is ax_in

    def test_components_horizontal(self, gnss_3comp, rect_fault):
        ax = geodef.plot.vectors(gnss_3comp, rect_fault, components="horizontal")
        assert isinstance(ax, plt.Axes)

    def test_components_vertical(self, gnss_3comp, rect_fault):
        ax = geodef.plot.vectors(gnss_3comp, rect_fault, components="vertical")
        assert isinstance(ax, plt.Axes)

    def test_no_ellipses(self, gnss_3comp, rect_fault):
        ax = geodef.plot.vectors(gnss_3comp, rect_fault, ellipses=False)
        assert isinstance(ax, plt.Axes)

    def test_scale_factor(self, gnss_3comp, rect_fault):
        ax = geodef.plot.vectors(gnss_3comp, rect_fault, scale=500.0)
        assert isinstance(ax, plt.Axes)

    def test_custom_colors(self, gnss_3comp, rect_fault):
        ax = geodef.plot.vectors(gnss_3comp, rect_fault,
                                  obs_color="blue", pred_color="green")
        assert isinstance(ax, plt.Axes)

    def test_with_legend(self, gnss_3comp, rect_fault):
        predicted = gnss_3comp.obs * 0.9
        ax = geodef.plot.vectors(gnss_3comp, rect_fault,
                                  predicted=predicted, legend=True)
        legend = ax.get_legend()
        assert legend is not None

    def test_quiver_kwargs(self, gnss_3comp, rect_fault):
        ax = geodef.plot.vectors(gnss_3comp, rect_fault,
                                  quiver_kwargs={"width": 0.005})
        assert isinstance(ax, plt.Axes)

    def test_ellipse_kwargs(self, gnss_3comp, rect_fault):
        ax = geodef.plot.vectors(gnss_3comp, rect_fault,
                                  ellipse_kwargs={"alpha": 0.5})
        assert isinstance(ax, plt.Axes)

    def test_equal_aspect(self, gnss_3comp, rect_fault):
        ax = geodef.plot.vectors(gnss_3comp, rect_fault)
        assert ax.get_aspect() in ("equal", 1.0)

    def test_vertical_only_from_3comp(self, gnss_3comp, rect_fault):
        ax = geodef.plot.vectors(gnss_3comp, rect_fault, components="vertical")
        assert isinstance(ax, plt.Axes)

    def test_vertical_not_available(self, gnss_horiz, rect_fault):
        with pytest.raises(ValueError, match="[Vv]ertical"):
            geodef.plot.vectors(gnss_horiz, rect_fault, components="vertical")

    def test_ellipses_at_tip(self, gnss_3comp, rect_fault):
        """Ellipses should be centered at arrow tips, not bases."""
        from matplotlib.patches import Ellipse
        ax = geodef.plot.vectors(gnss_3comp, rect_fault, scale=500.0)
        ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
        assert len(ellipses) == gnss_3comp.n_stations
        # Check that the first ellipse center differs from the station
        from geodef.plot import _stations_to_local_km
        x_km, y_km = _stations_to_local_km(gnss_3comp, rect_fault)
        ell0_center = ellipses[0].center
        tip_x = x_km[0] + gnss_3comp._ve[0] * 500.0
        tip_y = y_km[0] + gnss_3comp._vn[0] * 500.0
        np.testing.assert_allclose(ell0_center, (tip_x, tip_y), atol=1e-10)

    def test_scale_arrow_legend(self, gnss_3comp, rect_fault):
        predicted = gnss_3comp.obs * 0.9
        ax = geodef.plot.vectors(gnss_3comp, rect_fault,
                                  predicted=predicted, scale=500.0,
                                  legend=True, scale_arrow=0.01,
                                  scale_arrow_label="10 mm observed")
        # Should have quiver artists for the scale arrows (uses ax.quiver)
        from matplotlib.quiver import Quiver
        quivers = [c for c in ax.get_children() if isinstance(c, Quiver)]
        # At least 3: obs data, pred data, and 1-2 scale arrows
        assert len(quivers) >= 3

    def test_scale_arrow_obs_only(self, gnss_3comp, rect_fault):
        ax = geodef.plot.vectors(gnss_3comp, rect_fault, scale=500.0,
                                  legend=True, scale_arrow=0.01)
        assert isinstance(ax, plt.Axes)

    def test_legend_proxy_artists(self, gnss_3comp, rect_fault):
        """Without scale_arrow, legend should use proxy Line2D artists."""
        predicted = gnss_3comp.obs * 0.9
        ax = geodef.plot.vectors(gnss_3comp, rect_fault,
                                  predicted=predicted, legend=True)
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 2  # "Observed" and "Predicted"

    def test_vertical_colorbar_shown(self, gnss_3comp, rect_fault):
        """Vertical mode should show colorbar by default."""
        ax = geodef.plot.vectors(gnss_3comp, rect_fault, components="vertical")
        # colorbar adds a new axes to the figure
        assert len(ax.figure.axes) >= 2

    def test_vertical_colorbar_off(self, gnss_3comp, rect_fault):
        """vertical_colorbar=False suppresses the colorbar."""
        ax = geodef.plot.vectors(gnss_3comp, rect_fault, components="vertical",
                                  vertical_colorbar=False)
        # Only the main axes, no colorbar axes
        assert len(ax.figure.axes) == 1

    def test_vertical_dot_size_scales(self, gnss_3comp, rect_fault):
        """scale parameter should affect dot sizes in vertical mode."""
        ax1 = geodef.plot.vectors(gnss_3comp, rect_fault,
                                   components="vertical", scale=1)
        ax2 = geodef.plot.vectors(gnss_3comp, rect_fault,
                                   components="vertical", scale=5)
        # Get the scatter PathCollection
        sc1 = [c for c in ax1.get_children()
               if isinstance(c, matplotlib.collections.PathCollection)
               and len(c.get_offsets()) > 0][0]
        sc2 = [c for c in ax2.get_children()
               if isinstance(c, matplotlib.collections.PathCollection)
               and len(c.get_offsets()) > 0][0]
        # Larger scale should give larger max dot size
        assert sc2.get_sizes().max() > sc1.get_sizes().max()

    def test_scale_arrow_loc(self, gnss_3comp, rect_fault):
        """scale_arrow_loc should not raise for valid locations."""
        for loc in ("lower right", "lower left", "upper right", "upper left"):
            ax = geodef.plot.vectors(gnss_3comp, rect_fault, scale=500.0,
                                      legend=True, scale_arrow=0.01,
                                      scale_arrow_loc=loc)
            assert isinstance(ax, plt.Axes)
            plt.close(ax.figure)


# ======================================================================
# plot.insar (6.7d)
# ======================================================================

class TestPlotInSAR:
    """Tests for geodef.plot.insar."""

    def test_returns_axes_single(self, insar_data, rect_fault):
        ax = geodef.plot.insar(insar_data, rect_fault)
        assert isinstance(ax, plt.Axes)

    def test_layout_obs(self, insar_data, rect_fault):
        ax = geodef.plot.insar(insar_data, rect_fault, layout="obs")
        assert isinstance(ax, plt.Axes)

    def test_layout_obs_pred_res(self, insar_data, rect_fault):
        predicted = insar_data.obs * 0.9
        result = geodef.plot.insar(insar_data, rect_fault,
                                    predicted=predicted, layout="obs_pred_res")
        # Returns array of axes for multi-panel
        assert hasattr(result, "__len__")
        assert len(result) == 3

    def test_layout_residual(self, insar_data, rect_fault):
        predicted = insar_data.obs * 0.9
        ax = geodef.plot.insar(insar_data, rect_fault,
                                predicted=predicted, layout="residual")
        assert isinstance(ax, plt.Axes)

    def test_layout_pred(self, insar_data, rect_fault):
        predicted = insar_data.obs * 0.9
        ax = geodef.plot.insar(insar_data, rect_fault,
                                predicted=predicted, layout="pred")
        assert isinstance(ax, plt.Axes)

    def test_existing_axes(self, insar_data, rect_fault):
        fig, ax_in = plt.subplots()
        ax_out = geodef.plot.insar(insar_data, rect_fault, ax=ax_in)
        assert ax_out is ax_in

    def test_custom_cmap(self, insar_data, rect_fault):
        ax = geodef.plot.insar(insar_data, rect_fault, cmap="coolwarm")
        assert isinstance(ax, plt.Axes)

    def test_vmin_vmax(self, insar_data, rect_fault):
        ax = geodef.plot.insar(insar_data, rect_fault, vmin=-0.1, vmax=0.1)
        assert isinstance(ax, plt.Axes)

    def test_scatter_kwargs(self, insar_data, rect_fault):
        ax = geodef.plot.insar(insar_data, rect_fault,
                                scatter_kwargs={"s": 5, "marker": "s"})
        assert isinstance(ax, plt.Axes)

    def test_colorbar(self, insar_data, rect_fault):
        ax = geodef.plot.insar(insar_data, rect_fault, colorbar=True)
        assert len(ax.figure.get_axes()) >= 2

    def test_no_colorbar(self, insar_data, rect_fault):
        ax = geodef.plot.insar(insar_data, rect_fault, colorbar=False)
        assert len(ax.figure.get_axes()) == 1

    def test_residual_requires_predicted(self, insar_data, rect_fault):
        with pytest.raises(ValueError, match="predicted"):
            geodef.plot.insar(insar_data, rect_fault, layout="residual")

    def test_equal_aspect(self, insar_data, rect_fault):
        ax = geodef.plot.insar(insar_data, rect_fault)
        assert ax.get_aspect() in ("equal", 1.0)

    def test_obs_pred_res_ylabel_suppressed(self, insar_data, rect_fault):
        """Middle and right panels should not have y-axis labels."""
        predicted = insar_data.obs * 0.9
        axes = geodef.plot.insar(insar_data, rect_fault,
                                  predicted=predicted, layout="obs_pred_res")
        # First panel keeps ylabel
        assert axes[0].get_ylabel() != ""
        # Second and third panels should have empty ylabel
        assert axes[1].get_ylabel() == ""
        assert axes[2].get_ylabel() == ""


# ======================================================================
# plot.fit (6.7g)
# ======================================================================

class TestPlotFit:
    """Tests for geodef.plot.fit."""

    def test_scatter(self, gnss_3comp):
        obs = gnss_3comp.obs
        predicted = obs * 0.95
        ax = geodef.plot.fit(obs, predicted)
        assert isinstance(ax, plt.Axes)

    def test_one_to_one_line(self, gnss_3comp):
        obs = gnss_3comp.obs
        predicted = obs * 0.95
        ax = geodef.plot.fit(obs, predicted)
        # Should have scatter + 1:1 line
        assert len(ax.lines) >= 1

    def test_existing_axes(self, gnss_3comp):
        fig, ax_in = plt.subplots()
        obs = gnss_3comp.obs
        ax_out = geodef.plot.fit(obs, obs * 0.95, ax=ax_in)
        assert ax_out is ax_in

    def test_residual_histogram(self, gnss_3comp):
        obs = gnss_3comp.obs
        predicted = obs * 0.95
        ax = geodef.plot.fit(obs, predicted, style="residual_histogram")
        assert isinstance(ax, plt.Axes)
        # Should have histogram patches
        assert len(ax.patches) >= 1

    def test_kwargs_passthrough(self, gnss_3comp):
        obs = gnss_3comp.obs
        ax = geodef.plot.fit(obs, obs * 0.95, scatter_kwargs={"s": 10, "alpha": 0.5})
        assert isinstance(ax, plt.Axes)

    def test_invalid_style(self, gnss_3comp):
        obs = gnss_3comp.obs
        with pytest.raises(ValueError, match="style"):
            geodef.plot.fit(obs, obs * 0.95, style="invalid")


# ======================================================================
# plot.fault3d (6.7e)
# ======================================================================

class TestPlotFault3D:
    """Tests for geodef.plot.fault3d."""

    def test_returns_axes(self, rect_fault):
        ax = geodef.plot.fault3d(rect_fault)
        assert ax is not None
        # 3D axes
        from mpl_toolkits.mplot3d import Axes3D
        assert isinstance(ax, Axes3D)

    def test_color_by_depth(self, rect_fault):
        ax = geodef.plot.fault3d(rect_fault, color_by="depth")
        assert ax is not None

    def test_color_by_area(self, rect_fault):
        ax = geodef.plot.fault3d(rect_fault, color_by="area")
        assert ax is not None

    def test_color_by_array(self, rect_fault):
        values = np.random.rand(rect_fault.n_patches)
        ax = geodef.plot.fault3d(rect_fault, color_by=values)
        assert ax is not None

    def test_color_by_none(self, rect_fault):
        ax = geodef.plot.fault3d(rect_fault, color_by=None)
        assert ax is not None

    def test_existing_axes(self, rect_fault):
        fig = plt.figure()
        ax_in = fig.add_subplot(111, projection="3d")
        ax_out = geodef.plot.fault3d(rect_fault, ax=ax_in)
        assert ax_out is ax_in

    def test_tri_fault(self, tri_fault):
        ax = geodef.plot.fault3d(tri_fault, color_by="depth")
        assert ax is not None

    def test_station_overlay(self, rect_fault, gnss_3comp):
        ax = geodef.plot.fault3d(rect_fault, station_locations=gnss_3comp)
        assert ax is not None

    def test_kwargs_passthrough(self, rect_fault):
        ax = geodef.plot.fault3d(rect_fault, alpha=0.5)
        assert ax is not None

    def test_cmap(self, rect_fault):
        ax = geodef.plot.fault3d(rect_fault, cmap="plasma")
        assert ax is not None

    def test_view_angle(self, rect_fault):
        ax = geodef.plot.fault3d(rect_fault, view=(45, -120))
        assert ax is not None
        assert ax.elev == pytest.approx(45)
        assert ax.azim == pytest.approx(-120)

    def test_computed_zorder_disabled(self, rect_fault, gnss_3comp):
        """3D plot with stations should disable computed_zorder."""
        ax = geodef.plot.fault3d(rect_fault, station_locations=gnss_3comp)
        assert ax.computed_zorder is False


# ======================================================================
# plot.map (6.7f)
# ======================================================================

class TestPlotMap:
    """Tests for geodef.plot.map."""

    def test_returns_axes(self, rect_fault):
        ax = geodef.plot.map(rect_fault)
        assert isinstance(ax, plt.Axes)

    def test_with_datasets(self, rect_fault, gnss_3comp, insar_data):
        ax = geodef.plot.map(rect_fault, datasets=[gnss_3comp, insar_data])
        assert isinstance(ax, plt.Axes)

    def test_single_dataset(self, rect_fault, gnss_3comp):
        ax = geodef.plot.map(rect_fault, datasets=gnss_3comp)
        assert isinstance(ax, plt.Axes)

    def test_show_trace(self, rect_fault):
        ax = geodef.plot.map(rect_fault, show_trace=True)
        assert isinstance(ax, plt.Axes)
        # Should have at least patches + a line for the trace
        assert len(ax.collections) >= 1 or len(ax.lines) >= 1

    def test_no_patches(self, rect_fault):
        ax = geodef.plot.map(rect_fault, show_patches=False, show_trace=True)
        assert isinstance(ax, plt.Axes)

    def test_existing_axes(self, rect_fault):
        fig, ax_in = plt.subplots()
        ax_out = geodef.plot.map(rect_fault, ax=ax_in)
        assert ax_out is ax_in

    def test_equal_aspect(self, rect_fault):
        ax = geodef.plot.map(rect_fault)
        assert ax.get_aspect() in ("equal", 1.0)

    def test_patch_kwargs(self, rect_fault):
        ax = geodef.plot.map(rect_fault,
                              patch_kwargs={"edgecolor": "blue", "facecolor": "none"})
        assert isinstance(ax, plt.Axes)

    def test_trace_kwargs(self, rect_fault):
        ax = geodef.plot.map(rect_fault, show_trace=True,
                              trace_kwargs={"color": "red", "linewidth": 3})
        assert isinstance(ax, plt.Axes)

    def test_tri_fault(self, tri_fault):
        ax = geodef.plot.map(tri_fault)
        assert isinstance(ax, plt.Axes)

    def test_surface_trace_smooth(self, rect_fault):
        """Surface trace should be a smooth line, not a zigzag."""
        from geodef.plot import _get_surface_trace
        trace = _get_surface_trace(rect_fault)
        assert trace.shape[0] >= 2
        assert trace.shape[1] == 2
        # Project onto principal direction and check monotonic
        # (up to floating-point noise)
        centered = trace - trace.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ Vt[0]
        diffs = np.diff(proj)
        assert np.all(diffs >= -1e-3) or np.all(diffs <= 1e-3), \
            "Surface trace should be monotonic along principal direction"

    def test_map_with_values(self, rect_fault):
        """Map should color patches when values array is provided."""
        vals = np.random.rand(rect_fault.n_patches)
        ax = geodef.plot.map(rect_fault, values=vals, cmap="hot",
                              colorbar_label="Test")
        assert isinstance(ax, plt.Axes)
        # Colorbar adds an extra axes
        assert len(ax.figure.axes) >= 2

    def test_map_with_slip_vector(self, rect_fault, slip_magnitude):
        """Map should accept slip_vector and decompose it."""
        ax = geodef.plot.map(rect_fault, slip_vector=slip_magnitude,
                              component="magnitude",
                              colorbar_label="Slip (m)")
        assert isinstance(ax, plt.Axes)

    def test_map_values_and_slip_exclusive(self, rect_fault, slip_magnitude):
        """Providing both values and slip_vector should raise."""
        vals = np.random.rand(rect_fault.n_patches)
        with pytest.raises(ValueError, match="not both"):
            geodef.plot.map(rect_fault, values=vals,
                            slip_vector=slip_magnitude)

    def test_map_no_colorbar(self, rect_fault):
        """colorbar=False should suppress colorbar even with values."""
        vals = np.random.rand(rect_fault.n_patches)
        ax = geodef.plot.map(rect_fault, values=vals, colorbar=False)
        assert len(ax.figure.axes) == 1


# ======================================================================
# LCurveResult.plot / ABICCurveResult.plot refactor (6.7h)
# ======================================================================

class TestLCurvePlotRefactor:
    """Tests for refactored LCurveResult.plot."""

    def _make_lcurve(self):
        from geodef.invert import LCurveResult
        return LCurveResult(
            smoothing_values=np.logspace(-2, 2, 10),
            misfits=np.logspace(1, -1, 10),
            model_norms=np.logspace(-1, 1, 10),
            optimal=1.0,
        )

    def test_returns_axes(self):
        lc = self._make_lcurve()
        ax = lc.plot()
        assert isinstance(ax, plt.Axes)

    def test_existing_axes(self):
        lc = self._make_lcurve()
        fig, ax_in = plt.subplots()
        ax_out = lc.plot(ax=ax_in)
        assert ax_out is ax_in

    def test_kwargs(self):
        lc = self._make_lcurve()
        ax = lc.plot(marker_kwargs={"color": "green", "markersize": 15},
                      line_kwargs={"color": "purple", "linewidth": 3})
        assert isinstance(ax, plt.Axes)

    def test_annotate_default(self):
        """By default, the optimal point should be annotated."""
        lc = self._make_lcurve()
        ax = lc.plot()
        texts = ax.texts
        assert any("λ" in t.get_text() for t in texts)

    def test_annotate_off(self):
        """annotate=False should suppress the label."""
        lc = self._make_lcurve()
        ax = lc.plot(annotate=False)
        texts = ax.texts
        assert not any("λ" in t.get_text() for t in texts)


class TestABICCurvePlotRefactor:
    """Tests for refactored ABICCurveResult.plot."""

    def _make_abic(self):
        from geodef.invert import ABICCurveResult
        return ABICCurveResult(
            smoothing_values=np.logspace(-2, 2, 10),
            abic_values=np.random.rand(10) * 100 + 50,
            misfits=np.logspace(1, -1, 10),
            model_norms=np.logspace(-1, 1, 10),
            optimal=1.0,
        )

    def test_returns_axes(self):
        ac = self._make_abic()
        ax = ac.plot()
        assert isinstance(ax, plt.Axes)

    def test_existing_axes(self):
        ac = self._make_abic()
        fig, ax_in = plt.subplots()
        ax_out = ac.plot(ax=ax_in)
        assert ax_out is ax_in

    def test_kwargs(self):
        ac = self._make_abic()
        ax = ac.plot(marker_kwargs={"color": "green"},
                      line_kwargs={"linewidth": 3})
        assert isinstance(ax, plt.Axes)

    def test_annotate_default(self):
        """By default, the optimal point should be annotated."""
        ac = self._make_abic()
        ax = ac.plot()
        texts = ax.texts
        assert any("λ" in t.get_text() for t in texts)

    def test_annotate_off(self):
        """annotate=False should suppress the label."""
        ac = self._make_abic()
        ax = ac.plot(annotate=False)
        texts = ax.texts
        assert not any("λ" in t.get_text() for t in texts)


# ======================================================================
# Module structure
# ======================================================================

class TestModuleStructure:
    """Tests for the plot module's public API."""

    def test_importable(self):
        import geodef.plot
        assert hasattr(geodef.plot, "patches")
        assert hasattr(geodef.plot, "slip")
        assert hasattr(geodef.plot, "resolution")
        assert hasattr(geodef.plot, "uncertainty")
        assert hasattr(geodef.plot, "vectors")
        assert hasattr(geodef.plot, "insar")
        assert hasattr(geodef.plot, "fit")
        assert hasattr(geodef.plot, "fault3d")
        assert hasattr(geodef.plot, "map")

    def test_accessible_from_geodef(self):
        assert hasattr(geodef, "plot")
        assert callable(geodef.plot.slip)
        assert callable(geodef.plot.vectors)
