"""Tests for gradient-based nonlinear geometry inversion.

Synthetic-recovery tests: forward-model a known fault geometry, then
recover selected geometry parameters from a perturbed starting point.
Skipped entirely when JAX is not installed.
"""

import numpy as np
import pytest

from geodef import backend
from geodef.data import GNSS
from geodef.fault import Fault
from geodef.geometry import LocalFrame
from geodef.invert import geometry_search

jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True)
def jax_backend():
    """Run every test in this module on the JAX backend."""
    backend.set_backend("jax")
    yield
    backend.set_backend("numpy")


_REF_LAT, _REF_LON = -2.0, 100.0
_TRUE = {
    "depth": 25e3,
    "strike": 315.0,
    "dip": 15.0,
    "length": 120e3,
    "width": 60e3,
}
_NL, _NW = 4, 3


def _make_fault(dip=None, depth=None):
    return Fault.planar(
        lat=_REF_LAT,
        lon=_REF_LON,
        depth=depth if depth is not None else _TRUE["depth"],
        strike=_TRUE["strike"],
        dip=dip if dip is not None else _TRUE["dip"],
        length=_TRUE["length"],
        width=_TRUE["width"],
        n_length=_NL,
        n_width=_NW,
    )


@pytest.fixture(scope="module")
def gnss_data():
    """Noise-free GNSS velocities from a smooth dip-slip distribution."""
    backend.set_backend("numpy")
    fault = _make_fault()
    n_patches = fault.n_patches
    i = np.arange(n_patches) % _NL
    j = np.arange(n_patches) // _NL
    bump = np.exp(-(((i - 1.5) / 1.5) ** 2 + (j - 1.0) ** 2))
    slip_ss = np.zeros(n_patches)
    slip_ds = 3.0 * bump

    glon, glat = np.meshgrid(np.linspace(99.0, 101.0, 5), np.linspace(-3.0, -1.0, 5))
    glon, glat = glon.ravel(), glat.ravel()
    ue, un, uz = fault.displacement(glat, glon, slip_ss, slip_ds)
    n = len(glat)
    sigma = 0.001
    data = GNSS(
        lon=glon,
        lat=glat,
        ve=ue,
        vn=un,
        vu=uz,
        se=np.full(n, sigma),
        sn=np.full(n, sigma),
        su=np.full(n, sigma),
    )
    backend.set_backend("jax")
    return data


_THETA0_KWARGS = dict(
    n_length=_NL,
    n_width=_NW,
    smoothing="laplacian",
    smoothing_strength=1.0,
    components="dip",
)


def _theta_start(**overrides):
    theta = np.array(
        [
            0.0,
            0.0,
            _TRUE["depth"],
            _TRUE["strike"],
            _TRUE["dip"],
            _TRUE["length"],
            _TRUE["width"],
        ]
    )
    names = ["e0", "n0", "depth", "strike", "dip", "length", "width"]
    for key, value in overrides.items():
        theta[names.index(key)] = value
    return theta


class TestGeometrySearch:
    def test_accepts_mapping_and_returns_fault(self, gnss_data):
        frame = LocalFrame(_REF_LAT, _REF_LON)
        parameters = dict(
            zip(
                ["e0", "n0", "depth", "strike", "dip", "length", "width"],
                _theta_start(dip=30.0),
                strict=True,
            )
        )

        result = geometry_search(
            parameters,
            gnss_data,
            frame=frame,
            free=["dip"],
            bounds={"dip": (5.0, 45.0)},
            **_THETA0_KWARGS,
        )

        assert isinstance(result.fault, Fault)
        assert result.fault.frame is frame
        assert result.frame is frame
        assert abs(np.mean(result.fault.dip) - _TRUE["dip"]) < 0.5

    def test_rejects_mapping_with_missing_parameter(self, gnss_data):
        with pytest.raises(ValueError, match="missing keys"):
            geometry_search(
                {"depth": 20_000.0},
                gnss_data,
                frame=LocalFrame(_REF_LAT, _REF_LON),
                free=["dip"],
                **_THETA0_KWARGS,
            )

    def test_recovers_dip(self, gnss_data):
        result = geometry_search(
            _theta_start(dip=30.0),
            gnss_data,
            ref_lat=_REF_LAT,
            ref_lon=_REF_LON,
            free=["dip"],
            bounds={"dip": (5.0, 45.0)},
            **_THETA0_KWARGS,
        )
        assert abs(result.theta[4] - _TRUE["dip"]) < 0.5
        assert result.success

    def test_recovers_dip_and_depth(self, gnss_data):
        result = geometry_search(
            _theta_start(dip=25.0, depth=35e3),
            gnss_data,
            ref_lat=_REF_LAT,
            ref_lon=_REF_LON,
            free=["dip", "depth"],
            bounds={"dip": (5.0, 45.0), "depth": (10e3, 60e3)},
            **_THETA0_KWARGS,
        )
        assert abs(result.theta[4] - _TRUE["dip"]) < 1.0
        assert abs(result.theta[2] - _TRUE["depth"]) < 3e3

    def test_result_contents(self, gnss_data):
        result = geometry_search(
            _theta_start(dip=20.0),
            gnss_data,
            ref_lat=_REF_LAT,
            ref_lon=_REF_LON,
            free=["dip"],
            **_THETA0_KWARGS,
        )
        assert result.theta.shape == (7,)
        assert result.free == ["dip"]
        assert result.slip.shape == (_NL * _NW,)  # components='dip'
        assert result.theta_cov.shape == (1, 1)
        assert result.theta_cov[0, 0] >= 0.0
        assert result.chi2 >= 0.0
        assert np.isfinite(result.reduced_chi2)

    def test_bounds_respected(self, gnss_data):
        result = geometry_search(
            _theta_start(dip=30.0),
            gnss_data,
            ref_lat=_REF_LAT,
            ref_lon=_REF_LON,
            free=["dip"],
            bounds={"dip": (25.0, 45.0)},  # excludes the true dip of 15
            **_THETA0_KWARGS,
        )
        assert result.theta[4] >= 25.0 - 1e-9

    def test_requires_jax_backend(self, gnss_data):
        backend.set_backend("numpy")
        with pytest.raises(RuntimeError, match="JAX backend"):
            geometry_search(
                _theta_start(),
                gnss_data,
                ref_lat=_REF_LAT,
                ref_lon=_REF_LON,
                free=["dip"],
                **_THETA0_KWARGS,
            )

    def test_unknown_free_parameter_raises(self, gnss_data):
        with pytest.raises(ValueError, match="rake"):
            geometry_search(
                _theta_start(),
                gnss_data,
                ref_lat=_REF_LAT,
                ref_lon=_REF_LON,
                free=["rake"],
                **_THETA0_KWARGS,
            )
