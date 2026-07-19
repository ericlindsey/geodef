"""Green's-function engine registry and capability declarations.

Private module (roadmap 3.3). Each engine — a source geometry plus its
kernel family — is described once by an :class:`EngineSpec`; ``Fault``
and the assembly layer look capabilities up here instead of branching on
``engine == ...`` strings at every call site. The registry is private:
public registration waits until at least two external engines exercise
the callable contract (roadmap 3.3 / Phase 6.2).

The Green's callables share one signature::

    displacement_greens(fault, obs_lat, obs_lon, *, nu) -> G_raw
    strain_greens(fault, obs_lat, obs_lon, *, nu, obs_depth) -> G_raw

with ``G_raw`` shaped ``(n_comp * M, 2 * N)`` in the blocked
strike/dip column convention. ``None`` for a callable declares the
capability absent, and :func:`require` turns that into an actionable
error naming the engines that do support it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from geodef import greens as _greens

if TYPE_CHECKING:
    from geodef.fault import Fault

_GreensFn = Callable[..., np.ndarray]


@dataclasses.dataclass(frozen=True)
class EngineSpec:
    """Capability declaration for one Green's-function engine.

    Attributes:
        name: Registry key; the public ``Fault.engine`` string.
        geometry: Source geometry kind, ``"rect"`` or ``"tri"``.
        displacement_greens: Surface-displacement Green's assembly, or
            ``None`` if unsupported.
        strain_greens: Strain Green's assembly (surface or at depth via
            ``obs_depth``), or ``None`` if unsupported.
        surface_displacement: Whether z = 0 displacement is supported.
        internal_strain: Whether strain below the surface is supported.
        autodiff: Whether :mod:`geodef.gradients` has a differentiable
            forward model for this engine.
    """

    name: str
    geometry: str
    displacement_greens: _GreensFn | None
    strain_greens: _GreensFn | None
    surface_displacement: bool
    internal_strain: bool
    autodiff: bool


_REGISTRY: dict[str, EngineSpec] = {}


def register(spec: EngineSpec) -> None:
    """Register an engine spec under its name (replacing any previous)."""
    _REGISTRY[spec.name] = spec


def get(name: str) -> EngineSpec:
    """Look up an engine by name.

    Raises:
        ValueError: If no engine with that name is registered, naming
            the registered engines.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown engine: {name!r}. Registered engines: {names()}"
        ) from None


def names() -> tuple[str, ...]:
    """Names of all registered engines, sorted."""
    return tuple(sorted(_REGISTRY))


def require(spec: EngineSpec, capability: str) -> _GreensFn:
    """Return the requested Green's callable or raise an actionable error.

    Args:
        spec: The engine spec to query.
        capability: ``"displacement_greens"`` or ``"strain_greens"``.

    Raises:
        ValueError: If the engine does not provide the capability,
            naming the engines that do.
    """
    fn: _GreensFn | None = getattr(spec, capability)
    if fn is not None:
        return fn
    supported = tuple(
        s.name for _, s in sorted(_REGISTRY.items()) if getattr(s, capability)
    )
    raise ValueError(
        f"Engine {spec.name!r} does not support {capability!r}; "
        f"engines that do: {supported}"
    )


def _rect_displacement_greens(
    fault: "Fault", obs_lat: np.ndarray, obs_lon: np.ndarray, *, nu: float
) -> np.ndarray:
    assert fault._length is not None and fault._width is not None
    return _greens.displacement_greens(
        obs_lat,
        obs_lon,
        fault._lat,
        fault._lon,
        fault._depth,
        fault.strike,
        fault.dip,
        fault._length,
        fault._width,
        nu=nu,
    )


def _rect_strain_greens(
    fault: "Fault",
    obs_lat: np.ndarray,
    obs_lon: np.ndarray,
    *,
    nu: float,
    obs_depth: np.ndarray | None,
) -> np.ndarray:
    assert fault._length is not None and fault._width is not None
    return _greens.strain_greens(
        obs_lat,
        obs_lon,
        fault._lat,
        fault._lon,
        fault._depth,
        fault.strike,
        fault.dip,
        fault._length,
        fault._width,
        nu=nu,
        obs_depth=obs_depth,
    )


def _tri_displacement_greens(
    fault: "Fault", obs_lat: np.ndarray, obs_lon: np.ndarray, *, nu: float
) -> np.ndarray:
    assert fault._vertices is not None
    return _greens.tri_displacement_greens(
        obs_lat,
        obs_lon,
        fault._lat,
        fault._lon,
        fault._depth,
        fault._vertices,
        nu=nu,
        frame=fault._frame,
    )


def _tri_strain_greens(
    fault: "Fault",
    obs_lat: np.ndarray,
    obs_lon: np.ndarray,
    *,
    nu: float,
    obs_depth: np.ndarray | None,
) -> np.ndarray:
    assert fault._vertices is not None
    return _greens.tri_strain_greens(
        obs_lat,
        obs_lon,
        fault._lat,
        fault._lon,
        fault._depth,
        fault._vertices,
        nu=nu,
        obs_depth=obs_depth,
        frame=fault._frame,
    )


register(
    EngineSpec(
        name="okada",
        geometry="rect",
        displacement_greens=_rect_displacement_greens,
        strain_greens=_rect_strain_greens,
        surface_displacement=True,
        internal_strain=True,
        autodiff=True,
    )
)

register(
    EngineSpec(
        name="tri",
        geometry="tri",
        displacement_greens=_tri_displacement_greens,
        strain_greens=_tri_strain_greens,
        surface_displacement=True,
        internal_strain=True,
        autodiff=True,
    )
)
