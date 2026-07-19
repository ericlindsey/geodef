"""Warped triangular-mesh geometry: TriWarp and its collapsed posterior.

Private submodule of :mod:`geodef.bayes`. ``TriWarp`` parameterizes
normal-offset warps of a triangular mesh with a low-dimensional knot
field; ``TriPosterior`` runs the collapsed formulation over those knots.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.linalg

from geodef import backend
from geodef.bayes._collapsed import _CollapsedPosterior
from geodef.bayes._util import _VALID_MODES, _require_jax
from geodef.data import DataSet
from geodef.fault import Fault
from geodef.gradients import tri_greens
from geodef.invert import LinearSystem
from geodef.invert._geometry import _projection_matrix
from geodef.invert._solvers import _rank_positive_eigs


class TriWarp:
    """Low-dimensional normal-offset parameterization of a triangular mesh.

    Warps a fixed-connectivity triangular mesh along its best-fit-plane
    normal by smoothly interpolating offsets (meters) from a small set of
    control knots with a Gaussian RBF. Connectivity and the interpolation
    matrix are frozen at construction, so ``vertices(theta)`` is an exact
    linear map of ``theta`` — no remeshing, hence no posterior
    discontinuities from connectivity flips, and every warped mesh is
    automatically watertight (vertices that coincide before warping share
    the same interpolated offset, since they share the same (u, v)).

    Args:
        fault: Reference triangular :class:`Fault`; kept for :meth:`fault` and
            by :class:`TriPosterior`.
        knots: Explicit knot locations in the mesh's best-fit-plane (u, v)
            coordinates, shape (nk, 2). Takes precedence over ``n_knots``.
        n_knots: ``(n_u, n_v)`` grid shape spanning the mesh's (u, v)
            bounding box (corners included) when ``knots`` is None. Flat
            knot index ``k = iu + n_u * iv`` (v varies slowest).
        length_scale: Gaussian RBF length scale in meters. Defaults to
            the larger of the two knot-grid spacings for a grid, or the
            median nearest-neighbor knot distance for explicit knots.
        ridge: Diagonal ridge added to the knot kernel matrix before
            solving for the interpolation weights (numerical stability).

    Raises:
        ValueError: If ``fault`` is not triangular, ``knots`` has the wrong
            shape, or no default ``length_scale`` can be inferred.
    """

    def __init__(
        self,
        fault: Fault,
        *,
        knots: npt.ArrayLike | None = None,
        n_knots: tuple[int, int] = (3, 2),
        length_scale: float | None = None,
        ridge: float = 1e-8,
    ) -> None:
        if fault.vertices is None:
            raise ValueError(
                "TriWarp requires a triangular Fault (fault.vertices is None)"
            )
        self._ref_fault = fault
        self.frame = fault.frame
        self._ref_lat = self.frame.origin_lat
        self._ref_lon = self.frame.origin_lon

        v0 = np.asarray(fault.vertices, dtype=float)
        self._shape = v0.shape
        self._n_tri = v0.shape[0]
        p = v0.reshape(-1, 3)
        self._v0_flat = p

        # Best-fit plane by SVD: e_u, n_hat from the first/last right
        # singular vectors; n_hat flipped to point up, e_v recomputed so
        # (e_u, e_v, n_hat) is right-handed.
        center = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - center)
        e_u, n_hat = vt[0], vt[2]
        if n_hat[2] < 0.0:
            n_hat = -n_hat
        e_v = np.cross(n_hat, e_u)
        self._center = center
        self._e_u = e_u
        self._e_v = e_v
        self._n_hat = n_hat

        uv = (p - center) @ np.column_stack([e_u, e_v])

        if knots is not None:
            knots_uv = np.asarray(knots, dtype=float)
            if knots_uv.ndim != 2 or knots_uv.shape[1] != 2:
                raise ValueError("knots must have shape (nk, 2)")
            diff = knots_uv[:, None, :] - knots_uv[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, np.inf)
            default_length_scale = float(np.median(dist.min(axis=1)))
        else:
            n_u, n_v = n_knots
            u_vals = np.linspace(uv[:, 0].min(), uv[:, 0].max(), n_u)
            v_vals = np.linspace(uv[:, 1].min(), uv[:, 1].max(), n_v)
            uu, vv = np.meshgrid(u_vals, v_vals)
            knots_uv = np.column_stack([uu.ravel(), vv.ravel()])
            du = (u_vals[-1] - u_vals[0]) / (n_u - 1) if n_u > 1 else 0.0
            dv = (v_vals[-1] - v_vals[0]) / (n_v - 1) if n_v > 1 else 0.0
            default_length_scale = max(du, dv)

        if length_scale is not None:
            self._length_scale = float(length_scale)
        elif default_length_scale > 0.0:
            self._length_scale = default_length_scale
        else:
            raise ValueError(
                "Could not infer a default length_scale from the knot "
                "layout; pass length_scale explicitly."
            )

        self.knots_uv = knots_uv
        self.knots_xyz = (
            center + np.outer(knots_uv[:, 0], e_u) + np.outer(knots_uv[:, 1], e_v)
        )
        self._n_knots = knots_uv.shape[0]

        two_l2 = 2.0 * self._length_scale**2
        kk_diff = knots_uv[:, None, :] - knots_uv[None, :, :]
        phi_kk = np.exp(-np.sum(kk_diff**2, axis=-1) / two_l2)
        a_mat = phi_kk + ridge * np.eye(self._n_knots)
        pk_diff = uv[:, None, :] - knots_uv[None, :, :]
        phi_pk = np.exp(-np.sum(pk_diff**2, axis=-1) / two_l2)
        self._b = scipy.linalg.solve(a_mat, phi_pk.T, assume_a="pos").T

    @property
    def n_knots(self) -> int:
        """Number of control knots."""
        return self._n_knots

    @property
    def length_scale(self) -> float:
        """Gaussian RBF length scale, in meters."""
        return self._length_scale

    @property
    def normal(self) -> np.ndarray:
        """Best-fit-plane unit normal, shape (3,), oriented upward."""
        return self._n_hat

    def offsets(self, theta: npt.ArrayLike) -> Any:
        """Per-flat-vertex normal offsets in meters, ``B @ theta``.

        Traceable — works under ``jit``/``vmap``/``jax.jacfwd`` on the
        JAX backend, and plain NumPy otherwise.

        Args:
            theta: Knot offsets, shape (n_knots,).

        Returns:
            Offsets, shape (3 * n_triangles,).
        """
        return backend.xp.asarray(self._b) @ backend.xp.asarray(theta)

    def vertices(self, theta: npt.ArrayLike) -> Any:
        """Warped mesh vertices, ``V0 + normal * offsets(theta)``.

        Traceable — works under ``jit``/``vmap``/``jax.jacfwd`` on the
        JAX backend, and plain NumPy otherwise.

        Args:
            theta: Knot offsets, shape (n_knots,).

        Returns:
            Vertex coordinates, shape (n_triangles, 3, 3).
        """
        off = self.offsets(theta)
        v_flat = backend.xp.asarray(self._v0_flat) + off[:, None] * backend.xp.asarray(
            self._n_hat
        )
        return backend.xp.reshape(v_flat, self._shape)

    def check(self, theta: npt.ArrayLike) -> bool:
        """True iff every warped vertex satisfies the half-space ``z <= 0``.

        Args:
            theta: Knot offsets, shape (n_knots,).

        Returns:
            Whether the warped mesh stays entirely underground.
        """
        off = backend.to_numpy(self.offsets(np.asarray(theta, dtype=float)))
        v_flat = self._v0_flat + off[:, None] * self._n_hat
        return bool(np.all(v_flat[:, 2] <= 0.0))

    def fault(self, theta: npt.ArrayLike) -> Fault:
        """Build a concrete ``Fault`` at warp parameters ``theta``.

        Uses the reference fault's (mean-centroid) frame, so the result
        is a drop-in replacement for the reference fault in any existing
        forward-modeling or plotting tool.

        Args:
            theta: Knot offsets, shape (n_knots,).

        Returns:
            A triangular ``Fault`` with the warped geometry.
        """
        verts = backend.to_numpy(self.vertices(np.asarray(theta, dtype=float)))
        return Fault.from_triangles(
            verts.astype(float),
            frame=self.frame,
            medium=self._ref_fault.medium,
        )

    def plot(self, theta: npt.ArrayLike | None = None, ax: Any = None) -> tuple:
        """3D preview of the reference mesh, an optional warp, and knots.

        A quick way to sanity-check a warp's geometry — and set sensible
        ``knot_prior`` bounds via :meth:`check` — before handing it to
        :class:`TriPosterior`.

        Args:
            theta: Warp parameters to preview, or None to show only the
                reference mesh.
            ax: Existing 3D matplotlib axes to draw on; a new figure and
                axes are created when None.

        Returns:
            Tuple ``(fig, ax)``.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
        else:
            fig = ax.figure

        ref_tris = self._v0_flat.reshape(self._shape)
        wire = Poly3DCollection(
            ref_tris, facecolors="none", edgecolors="0.6", linewidths=0.6
        )
        ax.add_collection3d(wire)
        all_pts = ref_tris.reshape(-1, 3)

        if theta is not None:
            theta_arr = np.asarray(theta, dtype=float)
            verts = backend.to_numpy(self.vertices(theta_arr))
            per_vertex = backend.to_numpy(self.offsets(theta_arr)).reshape(
                self._n_tri, 3
            )
            tri_offset = per_vertex.mean(axis=1)
            vmax = float(np.max(np.abs(tri_offset))) or 1.0
            cmap = plt.get_cmap("coolwarm")
            colors = cmap(0.5 + 0.5 * tri_offset / vmax)
            warped = Poly3DCollection(
                verts, facecolors=colors, edgecolors="k", linewidths=0.3, alpha=0.9
            )
            ax.add_collection3d(warped)
            all_pts = np.concatenate([all_pts, verts.reshape(-1, 3)])

        kx = self.knots_xyz
        ax.scatter(kx[:, 0], kx[:, 1], kx[:, 2], color="k", s=25, depthshade=False)
        for i in range(self._n_knots):
            ax.text(kx[i, 0], kx[i, 1], kx[i, 2], f"k{i}")

        pts = np.concatenate([all_pts, kx])
        center = pts.mean(axis=0)
        radius = float(np.max(np.linalg.norm(pts - center, axis=1))) or 1.0
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_zlabel("Up (m)")
        return fig, ax


def _parse_knot_prior(
    knot_prior: tuple | Sequence[tuple], n_knots: int, parse: Any
) -> list[tuple]:
    """Normalize a ``TriPosterior`` ``knot_prior`` to ``n_knots`` specs.

    Accepts a single ``(lo, hi)`` / ``('normal', mu, sd)`` spec applied to
    every knot, or a sequence of exactly ``n_knots`` per-knot specs.

    Args:
        knot_prior: The user-supplied prior spec(s).
        n_knots: Number of knots (``warp.n_knots``).
        parse: ``_CollapsedPosterior._parse_prior``, injected to avoid an
            import cycle.

    Returns:
        List of ``n_knots`` parsed specs.

    Raises:
        ValueError: If a sequence of per-knot specs has the wrong length.
    """
    is_single = (
        len(knot_prior) == 2 and all(isinstance(v, (int, float)) for v in knot_prior)
    ) or (len(knot_prior) == 3 and knot_prior[0] == "normal")
    if is_single:
        return [parse(f"knot{i}", knot_prior) for i in range(n_knots)]
    if len(knot_prior) != n_knots:
        raise ValueError(
            f"knot_prior sequence must have length warp.n_knots ({n_knots}), "
            f"got {len(knot_prior)}"
        )
    return [parse(f"knot{i}", spec) for i, spec in enumerate(knot_prior)]


class TriPosterior(_CollapsedPosterior):
    """Collapsed log-posterior for a warped triangular-mesh geometry.

    Uses the same collapsed slip machinery as :class:`RectPosterior` —
    the slip prior stays Gaussian, so slip is marginalized analytically
    — but the forward model is a linear normal-offset warp
    (:class:`TriWarp`) of a fixed-connectivity triangular mesh instead of
    a rectangular patch grid. There is no positivity path here (future
    work).

    The sampled parameter vector ``x`` stacks the warp's knot offsets
    (meters, ``warp.n_knots`` of them), then ``log10_sigma``, then
    ``log10_lambda`` when ``mode='hierarchical'``.

    Args:
        warp: A :class:`TriWarp` built from the reference triangular
            mesh; ``warp.vertices(theta)`` supplies the forward geometry.
        datasets: One or more displacement datasets (GNSS, InSAR,
            Vertical).
        knot_prior: Prior for every knot offset: a single ``(lo, hi)``
            or ``('normal', mu, sd)`` applied to all knots, or a sequence
            of ``warp.n_knots`` per-knot specs.
        knots0: Initial knot offsets (meters), shape (n_knots,) — the
            sampler's starting point, clipped into uniform prior bounds.
            Defaults to zeros (the reference mesh). Start from your best
            estimate when you have one: with tight data and a start far
            from the mode, the huge initial misfit makes the posterior
            extremely stiff in ``log10_sigma`` and warmup can fail to
            adapt (the same reason ``RectPosterior`` starts at
            ``theta0``).
        components: Slip components for the marginalized linear solve:
            ``'both'``, ``'strike'``, or ``'dip'``.
        mode: Slip-prior mode: ``'hierarchical'``, ``'weak'``, or
            ``'profiled'`` (see the module docstring).
        regularization: Regularization operator for the hierarchical and
            profiled modes; must be None for ``'weak'``.
        regularization_strength: Fixed lambda for ``'profiled'``; initial
            lambda (sampler starting point) for ``'hierarchical'``.
        slip_scale: Prior slip scale in meters for ``'weak'``.
        log10_sigma_prior: Uniform prior bounds on ``log10_sigma``.
        log10_lambda_prior: Uniform prior bounds on ``log10_lambda``
            (hierarchical mode only).
        nu: Poisson's ratio.

    Raises:
        RuntimeError: If the JAX backend is not active.
        ValueError: If ``knot_prior``, ``mode``, or ``components`` is
            invalid or inconsistent.
    """

    def __init__(
        self,
        warp: TriWarp,
        datasets: DataSet | list[DataSet],
        *,
        knot_prior: tuple | Sequence[tuple],
        knots0: npt.ArrayLike | None = None,
        components: str = "both",
        mode: str = "hierarchical",
        regularization: str | np.ndarray | None = "laplacian",
        regularization_strength: float | None = None,
        slip_scale: float | None = None,
        log10_sigma_prior: tuple[float, float] = (-2.0, 2.0),
        log10_lambda_prior: tuple[float, float] = (-8.0, 8.0),
        nu: float = 0.25,
    ) -> None:
        _require_jax()
        if isinstance(datasets, DataSet):
            datasets = [datasets]
        if knots0 is not None:
            knots0 = np.asarray(knots0, dtype=float)
            if knots0.shape != (warp.n_knots,):
                raise ValueError(
                    f"knots0 must have shape ({warp.n_knots},), got {knots0.shape}"
                )
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        if components not in ("both", "strike", "dip"):
            raise ValueError(
                f"components must be 'both', 'strike', or 'dip', got {components!r}"
            )
        if mode == "weak":
            if slip_scale is None:
                raise ValueError("mode='weak' requires slip_scale (meters)")
            if regularization is not None:
                raise ValueError(
                    "mode='weak' uses an identity slip prior; "
                    "regularization must be None"
                )
        if mode == "hierarchical" and regularization is None:
            raise ValueError("mode='hierarchical' requires a regularization operator")
        if mode == "profiled" and regularization_strength is None:
            raise ValueError(
                "mode='profiled' requires a fixed regularization_strength (lambda)"
            )

        self.mode = mode
        self.warp = warp
        self.components = components
        self.datasets = datasets
        self._nu = float(nu)
        nk = warp.n_knots

        ref_fault = warp._ref_fault
        sys = LinearSystem(ref_fault, datasets, regularization, components)
        n_patches = ref_fault.n_patches
        self._col_start, self._col_stop = {
            "both": (0, 2 * n_patches),
            "strike": (0, n_patches),
            "dip": (n_patches, 2 * n_patches),
        }[components]
        n_slip = self._col_stop - self._col_start
        self._n_slip = n_slip
        self._n_patches = n_patches

        frame = warp.frame
        self.frame = frame
        e_parts, n_parts = [], []
        for ds in datasets:
            enu = frame.to_enu(
                lon=ds.lon,
                lat=ds.lat,
                alt=np.full(ds.n_stations, frame.origin_alt),
            )
            e_parts.append(enu[:, 0])
            n_parts.append(enu[:, 1])
        e_obs = np.concatenate(e_parts)
        n_obs = np.concatenate(n_parts)
        self._obs = np.column_stack([e_obs, n_obs, np.zeros_like(e_obs)])

        self._W_half = scipy.linalg.cholesky(sys.W, lower=False)
        self._W_half_P = self._W_half @ _projection_matrix(datasets)
        self._d_w = self._W_half @ sys.d
        self.n_data = len(self._d_w)

        # Slip-prior precision structure: identical to RectPosterior's.
        if mode == "weak":
            assert slip_scale is not None
            self._LtL = np.eye(n_slip)
            self._lambda_fixed: float | None = 1.0 / slip_scale**2
            self._logdet_rank = n_slip
            self._logdet_sum = 0.0
        else:
            if regularization is not None:
                self._LtL = sys.LtL
                eig = np.abs(np.linalg.eigvalsh(self._LtL))
            else:
                self._LtL = np.zeros((n_slip, n_slip))
                eig = np.zeros(n_slip)
            if mode == "profiled":
                assert regularization_strength is not None
                self._lambda_fixed = float(regularization_strength)
            else:
                self._lambda_fixed = None
            pos = _rank_positive_eigs(eig)
            self._logdet_rank = len(pos)
            self._logdet_sum = float(np.sum(np.log(pos)))
        self._include_logdet = mode != "profiled"

        # Sampled-parameter layout and priors: knot offsets, then hypers.
        specs = _parse_knot_prior(knot_prior, nk, self._parse_prior)
        k0 = np.zeros(nk) if knots0 is None else knots0
        x0 = [
            float(np.clip(k0[i], spec[1], spec[2]))
            if spec[0] == "uniform"
            else float(k0[i])
            for i, spec in enumerate(specs)
        ]
        self.param_names = [f"knot{i}" for i in range(nk)] + ["log10_sigma"]
        specs.append(("uniform",) + tuple(map(float, log10_sigma_prior)))
        x0.append(float(np.clip(0.0, *log10_sigma_prior)))
        if mode == "hierarchical":
            self.param_names.append("log10_lambda")
            specs.append(("uniform",) + tuple(map(float, log10_lambda_prior)))
            lam0 = (
                float(np.log10(regularization_strength))
                if regularization_strength
                else 0.5 * (log10_lambda_prior[0] + log10_lambda_prior[1])
            )
            x0.append(float(np.clip(lam0, *log10_lambda_prior)))
        self.x0 = np.array(x0)
        self.n_params = len(self.param_names)
        self.free = [f"knot{i}" for i in range(nk)]

        self._is_uniform = np.array([s[0] == "uniform" for s in specs])
        self._lo = np.array([s[1] if s[0] == "uniform" else -np.inf for s in specs])
        self._hi = np.array([s[2] if s[0] == "uniform" else np.inf for s in specs])
        self._mu = np.array([s[1] if s[0] == "normal" else 0.0 for s in specs])
        self._sd = np.array([s[2] if s[0] == "normal" else 1.0 for s in specs])

        self._logpdf_fn = self._build_logpdf()

    def _assemble(self, x: Any) -> tuple:
        """Marginalization ingredients at sampled parameters x (traceable).

        Warps the reference mesh via ``warp.vertices(theta)``, clipping
        vertex depths to the half-space (``z <= 0``) before the tri
        kernel sees them so leapfrog excursions outside the prior bounds
        never hit an undefined geometry — the actual half-space
        violation is instead penalized in :meth:`log_prior`.

        Returns:
            Tuple ``(sigma2, lam, G_w, chol_H, m_hat, S)``, see
            :meth:`_CollapsedPosterior._assemble`.
        """
        import jax.numpy as jnp

        x = self._clip(x)
        nk = self.warp.n_knots
        theta = x[:nk]
        sigma2 = 10.0 ** (2.0 * x[nk])
        if self._lambda_fixed is not None:
            lam = jnp.asarray(self._lambda_fixed)
        else:
            lam = 10.0 ** x[nk + 1]

        v = self.warp.vertices(theta)
        v = v.at[..., 2].set(jnp.minimum(v[..., 2], 0.0))
        g3 = tri_greens(v, self._obs, self._nu)
        g_w = (jnp.asarray(self._W_half_P) @ g3)[:, self._col_start : self._col_stop]
        chol_h, m_hat, s = self._collapse_terms(g_w, lam)
        return sigma2, lam, g_w, chol_h, m_hat, s

    def log_prior(self, x: npt.ArrayLike) -> np.ndarray:
        """Base prior plus a traceable half-space guard on the warp.

        Adds ``-inf`` when the (bound-clipped) sampled knots would push
        any vertex of the *unclipped* warp above ``z = 0`` — the guard
        checks the true geometry, unlike :meth:`_assemble`'s numerical
        z-clamp, which only keeps the kernel's inputs well-defined.

        Args:
            x: Sampled parameter vector, ordered as ``param_names``.

        Returns:
            Scalar log-prior; ``-inf`` outside uniform bounds or when the
            warp violates the half-space.
        """
        import jax.numpy as jnp

        base = super().log_prior(x)
        x_c = self._clip(x)
        theta = x_c[: self.warp.n_knots]
        max_z = jnp.max(self.warp.vertices(theta)[..., 2])
        guard = jnp.where(max_z > 0.0, -jnp.inf, 0.0)
        return cast(np.ndarray, base + guard)
