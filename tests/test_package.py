"""Tests for geodef package structure and imports (Phase 2.1).

Verifies that the package is importable, modules are accessible,
the okada dispatcher works correctly, and the top-level API is usable.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Package importability
# ---------------------------------------------------------------------------

class TestPackageImports:
    """Verify that geodef and its submodules are importable."""

    def test_import_geodef(self):
        import geodef
        assert hasattr(geodef, "__version__")

    def test_import_okada85(self):
        from geodef import okada85
        assert callable(okada85.displacement)
        assert callable(okada85.tilt)
        assert callable(okada85.strain)

    def test_import_okada92(self):
        from geodef import okada92
        assert callable(okada92.okada92)
        assert callable(okada92.DC3D)

    def test_import_tri(self):
        from geodef import tri
        assert callable(tri.TDdispFS)
        assert callable(tri.TDdispHS)
        assert callable(tri.TDstrainFS)
        assert callable(tri.TDstrainHS)

    def test_import_okada_dispatcher(self):
        from geodef import okada
        assert callable(okada.displacement)


# ---------------------------------------------------------------------------
# 2. Top-level convenience API
# ---------------------------------------------------------------------------

class TestTopLevelAPI:
    """Verify that key names are accessible directly from `import geodef`."""

    def test_version_string(self):
        import geodef
        assert isinstance(geodef.__version__, str)
        assert geodef.__version__ == "0.1.0"

    def test_dataset_classes_importable(self):
        import geodef
        assert hasattr(geodef, "DataSet")
        assert hasattr(geodef, "GNSS")
        assert hasattr(geodef, "InSAR")
        assert hasattr(geodef, "Vertical")

    def test_dataset_isinstance(self):
        import geodef
        import numpy as np
        lat = np.array([0.0])
        lon = np.array([100.0])
        g = geodef.GNSS(lat, lon, np.array([1.0]), np.array([0.5]),
                        np.array([-0.1]), np.array([0.1]),
                        np.array([0.1]), np.array([0.5]))
        assert isinstance(g, geodef.DataSet)

    def test_import_data_module(self):
        from geodef import data
        assert hasattr(data, "GNSS")
        assert hasattr(data, "InSAR")
        assert hasattr(data, "Vertical")
        assert hasattr(data, "DataSet")


# ---------------------------------------------------------------------------
# 3. Okada dispatcher tests
# ---------------------------------------------------------------------------

class TestOkadaDispatcher:
    """Test that okada.displacement auto-selects okada85 vs okada92."""

    @pytest.fixture
    def fault_params(self):
        """Standard fault parameters for dispatcher tests."""
        return dict(
            depth=5000.0,
            strike=0.0,
            dip=70.0,
            length=10000.0,
            width=5000.0,
            rake=0.0,
            slip=1.0,
            opening=0.0,
            nu=0.25,
        )

    def test_surface_observation_matches_okada85(self, fault_params):
        """When z=0 (surface), dispatcher should give okada85-identical results."""
        from geodef import okada, okada85

        e = np.array([5000.0, 10000.0, -3000.0])
        n = np.array([2000.0, -1000.0, 7000.0])

        # Dispatcher at z=0
        ue_d, un_d, uz_d = okada.displacement(
            e, n, z=0.0, **fault_params
        )

        # Direct okada85 (uses L/W parameter names)
        ue_85, un_85, uz_85 = okada85.displacement(
            e, n,
            depth=fault_params["depth"],
            strike=fault_params["strike"],
            dip=fault_params["dip"],
            L=fault_params["length"],
            W=fault_params["width"],
            rake=fault_params["rake"],
            slip=fault_params["slip"],
            open=fault_params["opening"],
            nu=fault_params["nu"],
        )

        np.testing.assert_array_equal(ue_d, ue_85)
        np.testing.assert_array_equal(un_d, un_85)
        np.testing.assert_array_equal(uz_d, uz_85)

    def test_depth_observation_uses_okada92(self, fault_params):
        """When z<0 (at depth), dispatcher must use okada92 and return results."""
        from geodef import okada

        e = np.array([5000.0])
        n = np.array([2000.0])

        ue, un, uz = okada.displacement(
            e, n, z=-1000.0, **fault_params
        )

        assert ue.shape == (1,)
        assert un.shape == (1,)
        assert uz.shape == (1,)
        assert np.all(np.isfinite(ue))

    def test_depth_observation_matches_okada92(self, fault_params):
        """Dispatcher at depth should match direct okada92 call."""
        from geodef import okada, okada92

        e_val = 5000.0
        n_val = 2000.0
        z_val = -1000.0
        p = fault_params

        # Direct okada92 call (scalar interface)
        disp_92, _ = okada92.okada92(
            e_val, n_val, z_val, p["depth"], p["strike"], p["dip"],
            p["length"], p["width"],
            p["slip"] * np.cos(np.radians(p["rake"])),
            p["slip"] * np.sin(np.radians(p["rake"])),
            p["opening"],
            G=1.0, nu=p["nu"],
        )

        # Dispatcher
        ue_d, un_d, uz_d = okada.displacement(
            np.array([e_val]), np.array([n_val]), z=z_val, **p
        )

        np.testing.assert_allclose(ue_d[0], disp_92[0, 0], rtol=1e-10)
        np.testing.assert_allclose(un_d[0], disp_92[1, 0], rtol=1e-10)
        np.testing.assert_allclose(uz_d[0], disp_92[2, 0], rtol=1e-10)

    def test_z_positional_matches_keyword(self, fault_params):
        """z as 3rd positional arg should match z as keyword."""
        from geodef import okada

        e = np.array([5000.0])
        n = np.array([2000.0])

        ue1, un1, uz1 = okada.displacement(e, n, 0.0, **fault_params)
        ue2, un2, uz2 = okada.displacement(e, n, z=0.0, **fault_params)

        np.testing.assert_array_equal(ue1, ue2)
        np.testing.assert_array_equal(un1, un2)
        np.testing.assert_array_equal(uz1, uz2)

    def test_scalar_z_broadcast(self, fault_params):
        """Scalar z should broadcast to all observation points."""
        from geodef import okada

        e = np.array([5000.0, 10000.0])
        n = np.array([2000.0, -1000.0])

        ue, un, uz = okada.displacement(e, n, z=-500.0, **fault_params)

        assert ue.shape == (2,)
        assert un.shape == (2,)
        assert uz.shape == (2,)

    def test_array_z_at_depth(self, fault_params):
        """Array z values (all at depth) should use okada92."""
        from geodef import okada

        e = np.array([5000.0, 10000.0])
        n = np.array([2000.0, -1000.0])
        z = np.array([-500.0, -1000.0])

        ue, un, uz = okada.displacement(e, n, z=z, **fault_params)

        assert ue.shape == (2,)
        assert np.all(np.isfinite(ue))

    def test_positive_z_raises(self, fault_params):
        """Positive z (above surface) should raise ValueError."""
        from geodef import okada

        e = np.array([5000.0])
        n = np.array([2000.0])

        with pytest.raises(ValueError, match="above.*surface|positive|z.*> 0"):
            okada.displacement(e, n, z=100.0, **fault_params)


# ---------------------------------------------------------------------------
# 4. Module-level function signatures preserved
# ---------------------------------------------------------------------------

class TestOkada85API:
    """Verify okada85 functions work through the new package."""

    def test_displacement_basic(self):
        from geodef import okada85

        ue, un, uz = okada85.displacement(
            e=np.array([10000.0]),
            n=np.array([0.0]),
            depth=5000.0,
            strike=0.0,
            dip=70.0,
            L=10000.0,
            W=5000.0,
            rake=0.0,
            slip=1.0,
            open=0.0,
        )
        assert ue.shape == (1,)
        assert np.all(np.isfinite(ue))

    def test_tilt_basic(self):
        from geodef import okada85

        uze, uzn = okada85.tilt(
            e=np.array([10000.0]),
            n=np.array([0.0]),
            depth=5000.0,
            strike=0.0,
            dip=70.0,
            L=10000.0,
            W=5000.0,
            rake=0.0,
            slip=1.0,
            open=0.0,
        )
        assert uze.shape == (1,)

    def test_strain_basic(self):
        from geodef import okada85

        unn, une, uen, uee = okada85.strain(
            e=np.array([10000.0]),
            n=np.array([0.0]),
            depth=5000.0,
            strike=0.0,
            dip=70.0,
            L=10000.0,
            W=5000.0,
            rake=0.0,
            slip=1.0,
            open=0.0,
        )
        assert unn.shape == (1,)


class TestOkada92API:
    """Verify okada92 functions work through the new package."""

    def test_okada92_basic(self):
        from geodef.okada92 import okada92

        disp, strain = okada92(
            X=10000.0, Y=0.0, Z=-1000.0,
            depth=5000.0, strike=0.0, dip=70.0,
            length=10000.0, width=5000.0,
            strike_slip=1.0, dip_slip=0.0, opening=0.0,
            G=1.0, nu=0.25,
        )
        assert disp.shape == (3, 1)
        assert strain.shape == (3, 3)

    def test_dc3d_accessible(self):
        from geodef.okada92 import DC3D
        assert callable(DC3D)


class TestTriAPI:
    """Verify tdcalc functions work through the new tri module."""

    @pytest.fixture
    def triangle_setup(self):
        """Simple triangle + observation points for testing."""
        tri = np.array([
            [-1000.0, 0.0, 0.0],
            [1000.0, 0.0, 0.0],
            [0.0, 0.0, -3000.0],
        ])
        obs = np.array([[5000.0, 5000.0, 0.0]])
        slip = np.array([1.0, 0.0, 0.0])
        nu = 0.25
        return obs, tri, slip, nu

    def test_TDdispHS(self, triangle_setup):
        from geodef import tri
        obs, triangle, slip, nu = triangle_setup
        result = tri.TDdispHS(obs, triangle, slip, nu)
        assert result.shape == (1, 3)

    def test_TDdispFS(self, triangle_setup):
        from geodef import tri
        obs, triangle, slip, nu = triangle_setup
        result = tri.TDdispFS(obs, triangle, slip, nu)
        assert result.shape == (1, 3)

    def test_TDstrainHS(self, triangle_setup):
        from geodef import tri
        obs, triangle, slip, nu = triangle_setup
        result = tri.TDstrainHS(obs, triangle, slip, nu)
        assert result.shape == (1, 6)

    def test_TDstrainFS(self, triangle_setup):
        from geodef import tri
        obs, triangle, slip, nu = triangle_setup
        result = tri.TDstrainFS(obs, triangle, slip, nu)
        assert result.shape == (1, 6)
