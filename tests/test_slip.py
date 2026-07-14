"""Tests for function-oriented slip conversions."""

import numpy as np
import numpy.testing as npt
import pytest

from geodef.fault import Fault
from geodef.slip import (
    from_azimuth,
    from_plate,
    from_rake,
    magnitude,
    pack,
    plate_rake_from_euler,
    rake,
    to_plate,
    unpack,
)


def test_pack_and_unpack_blocked_components() -> None:
    vector = pack(strike_slip=[1.0, 2.0], dip_slip=[3.0, 4.0])

    strike_slip, dip_slip = unpack(vector)

    npt.assert_allclose(vector, [1.0, 2.0, 3.0, 4.0])
    npt.assert_allclose(strike_slip, [1.0, 2.0])
    npt.assert_allclose(dip_slip, [3.0, 4.0])


def test_pack_rejects_mismatched_components() -> None:
    with pytest.raises(ValueError, match="same shape"):
        pack(strike_slip=[1.0, 2.0], dip_slip=[3.0])


def test_unpack_rejects_odd_vector() -> None:
    with pytest.raises(ValueError, match="even"):
        unpack([1.0, 2.0, 3.0])


def test_rake_conversion_returns_physical_components() -> None:
    strike_slip, dip_slip = from_rake([2.0, -3.0], rake_degrees=30.0)

    npt.assert_allclose(strike_slip, np.array([2.0, -3.0]) * np.cos(np.pi / 6))
    npt.assert_allclose(dip_slip, np.array([2.0, -3.0]) * 0.5)
    npt.assert_allclose(magnitude(strike_slip, dip_slip), [2.0, 3.0])
    npt.assert_allclose(rake(strike_slip, dip_slip), [30.0, -150.0])


def test_azimuth_conversion_uses_per_patch_strike() -> None:
    strike_slip, dip_slip = from_azimuth(
        [1.0, 2.0],
        azimuth_degrees=90.0,
        fault_strike_degrees=[0.0, 90.0],
    )

    npt.assert_allclose(strike_slip, [0.0, 2.0], atol=1e-15)
    npt.assert_allclose(dip_slip, [1.0, 0.0], atol=1e-15)


def test_plate_conversion_round_trip() -> None:
    strike_slip, dip_slip = from_plate(
        parallel=[1.0, 2.0],
        perpendicular=[3.0, 4.0],
        plate_rake_degrees=[0.0, 90.0],
    )

    parallel, perpendicular = to_plate(
        strike_slip,
        dip_slip,
        plate_rake_degrees=[0.0, 90.0],
    )

    npt.assert_allclose(strike_slip, [1.0, -4.0], atol=1e-15)
    npt.assert_allclose(dip_slip, [3.0, 2.0], atol=1e-15)
    npt.assert_allclose(parallel, [1.0, 2.0], atol=1e-15)
    npt.assert_allclose(perpendicular, [3.0, 4.0], atol=1e-15)


def test_euler_pole_defines_plate_rake() -> None:
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

    plate_rake = plate_rake_from_euler(fault, (90.0, 0.0, 1.0))

    npt.assert_allclose(plate_rake, 90.0, atol=0.2)
