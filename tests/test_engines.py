"""Engine registry and capability contracts (roadmap 3.3)."""

import numpy as np
import pytest

from geodef import _engines


def _dummy_greens(*args, **kwargs):
    return np.zeros((0, 0))


def test_builtin_engines_registered():
    assert _engines.names() == ("okada", "tri")


def test_builtin_capability_declarations():
    okada = _engines.get("okada")
    assert okada.geometry == "rect"
    assert okada.surface_displacement
    assert okada.internal_strain
    assert okada.autodiff
    tri = _engines.get("tri")
    assert tri.geometry == "tri"
    assert tri.surface_displacement
    assert tri.internal_strain
    assert tri.autodiff


def test_unknown_engine_error_names_registered_engines():
    with pytest.raises(ValueError, match=r"okada.*tri"):
        _engines.get("meade")


def test_require_returns_the_declared_callable():
    spec = _engines.get("okada")
    assert _engines.require(spec, "displacement_greens") is spec.displacement_greens
    assert _engines.require(spec, "strain_greens") is spec.strain_greens


def test_missing_capability_error_is_actionable():
    spec = _engines.EngineSpec(
        name="surface_only",
        geometry="rect",
        displacement_greens=_dummy_greens,
        strain_greens=None,
        surface_displacement=True,
        internal_strain=False,
        autodiff=False,
    )
    with pytest.raises(ValueError, match=r"surface_only.*strain_greens"):
        _engines.require(spec, "strain_greens")
    # ... and names the engines that do support the capability
    with pytest.raises(ValueError, match=r"okada"):
        _engines.require(spec, "strain_greens")


def test_registration_is_replace_and_restore(monkeypatch):
    monkeypatch.setitem(
        _engines._REGISTRY,
        "okada",
        _engines.EngineSpec(
            name="okada",
            geometry="rect",
            displacement_greens=_dummy_greens,
            strain_greens=None,
            surface_displacement=True,
            internal_strain=False,
            autodiff=False,
        ),
    )
    assert _engines.get("okada").strain_greens is None


def test_fault_greens_matrix_uses_registry(monkeypatch):
    """Fault.greens_matrix dispatches through the registered spec."""
    import geodef

    fault = geodef.Fault.planar(
        lat=0.0,
        lon=0.0,
        depth=10e3,
        strike=0.0,
        dip=30.0,
        length=20e3,
        width=10e3,
    )
    calls = []
    spec = _engines.get("okada")

    def recording(f, obs_lat, obs_lon, *, nu):
        calls.append(nu)
        return spec.displacement_greens(f, obs_lat, obs_lon, nu=nu)

    monkeypatch.setitem(
        _engines._REGISTRY,
        "okada",
        _engines.EngineSpec(
            name="okada",
            geometry="rect",
            displacement_greens=recording,
            strain_greens=spec.strain_greens,
            surface_displacement=True,
            internal_strain=True,
            autodiff=True,
        ),
    )
    G = fault.greens_matrix(np.array([0.3]), np.array([0.1]))
    assert calls == [fault.medium.poisson_ratio]
    assert G.shape == (3, 2)
