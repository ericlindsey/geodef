"""Tests for geodef.medium — elastic medium parameters and their threading."""

import numpy as np
import pytest

from geodef.fault import Fault
from geodef.medium import DEFAULT_MEDIUM, ElasticMedium

# ====================================================================
# ElasticMedium value object
# ====================================================================


class TestElasticMedium:
    def test_defaults(self) -> None:
        m = ElasticMedium()
        assert m.shear_modulus == 30e9
        assert m.poisson_ratio == 0.25

    def test_aliases(self) -> None:
        m = ElasticMedium(shear_modulus=40e9, poisson_ratio=0.3)
        assert m.mu == 40e9
        assert m.nu == 0.3

    def test_lame_lambda(self) -> None:
        m = ElasticMedium(shear_modulus=30e9, poisson_ratio=0.25)
        # lambda = 2 mu nu / (1 - 2 nu); Poisson solid has lambda == mu
        assert m.lame_lambda == pytest.approx(30e9)

    def test_immutable(self) -> None:
        m = ElasticMedium()
        with pytest.raises(AttributeError):
            m.shear_modulus = 1.0  # type: ignore[misc]

    def test_equality_and_hash(self) -> None:
        assert ElasticMedium() == ElasticMedium()
        assert hash(ElasticMedium()) == hash(ElasticMedium())
        assert ElasticMedium(poisson_ratio=0.3) != ElasticMedium()

    @pytest.mark.parametrize("bad_mu", [0.0, -1e9, float("nan"), float("inf")])
    def test_invalid_shear_modulus(self, bad_mu: float) -> None:
        with pytest.raises(ValueError, match="shear_modulus"):
            ElasticMedium(shear_modulus=bad_mu)

    @pytest.mark.parametrize("bad_nu", [-0.1, 0.5, 0.7, float("nan")])
    def test_invalid_poisson_ratio(self, bad_nu: float) -> None:
        with pytest.raises(ValueError, match="poisson_ratio"):
            ElasticMedium(poisson_ratio=bad_nu)

    def test_default_medium_singleton_value(self) -> None:
        assert DEFAULT_MEDIUM == ElasticMedium()


# ====================================================================
# Threading through Fault
# ====================================================================


@pytest.fixture
def fault_kwargs() -> dict:
    return dict(
        lat=0.0,
        lon=100.0,
        depth=15_000.0,
        strike=90.0,
        dip=30.0,
        length=40_000.0,
        width=20_000.0,
        n_length=2,
        n_width=2,
    )


class TestFaultMedium:
    def test_default_medium(self, fault_kwargs: dict) -> None:
        fault = Fault.planar(**fault_kwargs)
        assert fault.medium == DEFAULT_MEDIUM

    def test_explicit_medium(self, fault_kwargs: dict) -> None:
        medium = ElasticMedium(poisson_ratio=0.3)
        fault = Fault.planar(**fault_kwargs, medium=medium)
        assert fault.medium == medium

    def test_poisson_ratio_changes_greens(self, fault_kwargs: dict) -> None:
        """nu must actually reach the dislocation kernels."""
        obs_lat = np.array([0.05, 0.10])
        obs_lon = np.array([100.0, 100.1])
        G_default = Fault.planar(**fault_kwargs).greens_matrix(obs_lat, obs_lon)
        G_nu30 = Fault.planar(
            **fault_kwargs, medium=ElasticMedium(poisson_ratio=0.3)
        ).greens_matrix(obs_lat, obs_lon)
        assert not np.allclose(G_default, G_nu30)

    def test_moment_uses_medium(self, fault_kwargs: dict) -> None:
        medium = ElasticMedium(shear_modulus=40e9)
        fault = Fault.planar(**fault_kwargs, medium=medium)
        slip = np.ones(fault.n_patches)
        expected = 40e9 * np.sum(fault.areas)
        assert fault.moment(slip) == pytest.approx(expected)

    def test_moment_mu_override_still_works(self, fault_kwargs: dict) -> None:
        fault = Fault.planar(**fault_kwargs)
        slip = np.ones(fault.n_patches)
        assert fault.moment(slip, mu=60e9) == pytest.approx(2 * fault.moment(slip))

    def test_from_triangles_accepts_medium(self) -> None:
        vertices = np.array(
            [
                [[0.0, 0.0, 5000.0], [5000.0, 0.0, 5000.0], [0.0, 5000.0, 8000.0]],
            ]
        )
        medium = ElasticMedium(poisson_ratio=0.3)
        fault = Fault.from_triangles(
            ref_lat=0.0, ref_lon=100.0, vertices=vertices, medium=medium
        )
        assert fault.medium == medium


# ====================================================================
# Cache keys include the medium
# ====================================================================


class TestMediumCacheKeys:
    def test_greens_key_includes_nu(self, fault_kwargs: dict) -> None:
        from geodef.data import GNSS
        from geodef.greens import _build_greens_key

        gnss = GNSS(
            lon=np.array([100.0]),
            lat=np.array([0.05]),
            ve=np.array([0.01]),
            vn=np.array([0.01]),
            vu=None,
            se=np.array([0.001]),
            sn=np.array([0.001]),
            su=None,
        )
        from geodef.cache import compute_hash

        key_a = _build_greens_key(Fault.planar(**fault_kwargs), gnss)
        key_b = _build_greens_key(
            Fault.planar(**fault_kwargs, medium=ElasticMedium(poisson_ratio=0.3)),
            gnss,
        )
        assert compute_hash(key_a) != compute_hash(key_b)

    def test_stress_key_includes_medium(self, fault_kwargs: dict) -> None:
        from geodef.cache import compute_hash
        from geodef.fault import _build_stress_key

        fault_a = Fault.planar(**fault_kwargs)
        fault_b = Fault.planar(**fault_kwargs, medium=ElasticMedium(poisson_ratio=0.3))
        key_a = _build_stress_key(fault_a, 30e9)
        key_b = _build_stress_key(fault_b, 30e9)
        assert compute_hash(key_a) != compute_hash(key_b)
