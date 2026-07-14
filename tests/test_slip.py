"""Tests for canonical slip and displacement value objects."""

import numpy as np
import numpy.testing as npt
import pytest

from geodef import Displacement as PublicDisplacement
from geodef import SlipModel as PublicSlipModel
from geodef.fault import Fault
from geodef.slip import Displacement, SlipModel, plate_rake_from_euler


class TestSlipModel:
    """SlipModel keeps its physical components and model basis distinct."""

    def test_strike_dip_vector_is_blocked(self):
        model = SlipModel(strike=[1.0, 2.0], dip=[3.0, 4.0])

        assert model.basis == "strike_dip"
        assert model.n_patches == 2
        assert model.n_components == 2
        npt.assert_allclose(model.vector, [1.0, 2.0, 3.0, 4.0])
        npt.assert_allclose(model.strike, [1.0, 2.0])
        npt.assert_allclose(model.dip, [3.0, 4.0])

    def test_public_exports(self):
        assert PublicSlipModel is SlipModel
        assert PublicDisplacement is Displacement

    def test_fixed_rake_remains_one_component(self):
        model = SlipModel.from_rake([2.0, -3.0], rake=30.0)

        assert model.basis == "rake"
        assert model.n_components == 1
        npt.assert_allclose(model.vector, [2.0, -3.0])
        npt.assert_allclose(model.strike, np.array([2.0, -3.0]) * np.cos(np.pi / 6))
        npt.assert_allclose(model.dip, np.array([2.0, -3.0]) * 0.5)
        npt.assert_allclose(model.magnitude, [2.0, 3.0])

    def test_fixed_azimuth_uses_per_patch_strike(self):
        model = SlipModel.from_azimuth(
            [1.0, 2.0], azimuth=90.0, fault_strike=[0.0, 90.0]
        )

        assert model.basis == "azimuth"
        assert model.n_components == 1
        npt.assert_allclose(model.rake, [90.0, 0.0])
        npt.assert_allclose(model.strike, [0.0, 2.0], atol=1e-15)
        npt.assert_allclose(model.dip, [1.0, 0.0], atol=1e-15)

    def test_plate_coordinates_rotate_to_physical_components(self):
        model = SlipModel.from_plate_rake(
            parallel=[1.0, 2.0],
            perpendicular=[3.0, 4.0],
            plate_rake=[0.0, 90.0],
        )

        assert model.basis == "plate"
        assert model.n_components == 2
        npt.assert_allclose(model.vector, [1.0, 2.0, 3.0, 4.0])
        npt.assert_allclose(model.rake_parallel, [1.0, 2.0])
        npt.assert_allclose(model.rake_perpendicular, [3.0, 4.0])
        npt.assert_allclose(model.strike, [1.0, -4.0], atol=1e-15)
        npt.assert_allclose(model.dip, [3.0, 2.0], atol=1e-15)

    def test_euler_pole_defines_geographic_plate_direction(self):
        fault = Fault.planar(
            lat=0.0,
            lon=100.0,
            depth=10_000.0,
            strike=0.0,
            dip=30.0,
            length=20_000.0,
            width=10_000.0,
            n_length=2,
            n_width=1,
        )

        model = SlipModel.from_euler_pole(
            parallel=np.ones(2),
            perpendicular=np.zeros(2),
            fault=fault,
            pole=(90.0, 0.0, 1.0),
        )

        assert model.basis == "plate"
        assert model.plate_rake is not None
        npt.assert_allclose(model.plate_rake, 90.0, atol=0.2)
        npt.assert_allclose(model.strike, 0.0, atol=4e-3)
        npt.assert_allclose(model.dip, 1.0, atol=4e-3)

        rake = plate_rake_from_euler(fault, (90.0, 0.0, 1.0))
        npt.assert_allclose(rake, model.plate_rake)

    def test_arrays_are_copied_and_read_only(self):
        strike = np.array([1.0, 2.0])
        model = SlipModel(strike=strike, dip=[0.0, 0.0])
        strike[0] = 99.0

        assert model.strike[0] == 1.0
        with pytest.raises(ValueError):
            model.vector[0] = 5.0

    def test_rejects_mismatched_component_lengths(self):
        with pytest.raises(ValueError, match="same shape"):
            SlipModel(strike=[1.0, 2.0], dip=[3.0])


class TestDisplacement:
    """Displacement gives named fields without losing tuple idioms."""

    def test_named_fields_unpacking_and_vector(self):
        result = Displacement(east=[1.0, 2.0], north=[3.0, 4.0], up=[5.0, 6.0])

        east, north, up = result
        npt.assert_allclose(east, result.east)
        npt.assert_allclose(north, result.north)
        npt.assert_allclose(up, result.up)
        npt.assert_allclose(result.vector, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0])

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            Displacement(east=[1.0], north=[2.0, 3.0], up=[4.0])
