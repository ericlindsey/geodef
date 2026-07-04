"""Internal deformation due to a rectangular dislocation (Okada, 1992).

Vectorized Python port of Y. Okada's DC3D code (finite rectangular source
in a semi-infinite medium). The arithmetic keeps line-by-line
correspondence with the published Fortran, but all per-point control flow
(coordinate clamps, singular-term selection, the vertical-dip branch) is
expressed with ``where`` masks, so the kernel evaluates arrays of
observation points in one call and is trace-safe on the JAX backend.

Layout of the 12-component U vector, as in the Fortran reference:
U[0:3] displacements (UX, UY, UZ); U[3:6] x-derivatives (UXX, UYX, UZX);
U[6:9] y-derivatives (UXY, UYY, UZY); U[9:12] z-derivatives (UXZ, UYZ, UZZ).

Note: the earlier scalar port truncated the tensile-fault DU blocks (UA
was missing entries 10-12, UB 11-12, UC 8-12 in Fortran numbering), so
opening-mode strains silently omitted those gradient terms. This rewrite
restores the complete blocks from the published DC3D.f.
"""

import warnings
from typing import NamedTuple

import numpy as np

from geodef import backend
from geodef.backend import xp

F0, F1, F2, F3 = 0.0, 1.0, 2.0, 3.0
PI2 = 6.283185307179586
EPS = 1e-6


def okada92(X, Y, Z, depth, strike, dip, length, width,
            strike_slip, dip_slip, opening, G, nu, allow_singular=False):
    """Compute displacement and strain at depth due to a rectangular dislocation.

    Wrapper around DC3D (Okada, 1992) that accepts geographic coordinates
    (East, North, Up) relative to the fault centroid and returns results in
    the same geographic frame. Vectorized: X, Y, Z may be scalars or
    equal-length arrays of observation points.

    Args:
        X: Easting of observation point(s) relative to fault centroid.
        Y: Northing of observation point(s) relative to fault centroid.
        Z: Observation depth(s) (Z <= 0, with Z=0 at the free surface).
        depth: Depth of fault centroid (positive down).
        strike: Strike angle in degrees from North.
        dip: Dip angle in degrees from horizontal.
        length: Along-strike fault length.
        width: Down-dip fault width.
        strike_slip: Strike-slip dislocation (DISL1).
        dip_slip: Dip-slip dislocation (DISL2, positive = thrust).
        opening: Tensile dislocation (DISL3).
        G: Shear modulus.
        nu: Poisson's ratio.
        allow_singular: If True, return NaN for singular points instead of raising.

    Returns:
        Tuple of (displacement, strain):
        - displacement: shape (3, 1) for scalar input, (n, 3) for array
          input; [ue, un, uz] in geographic coords.
        - strain: shape (3, 3) for scalar input, (n, 3, 3) for array input;
          displacement gradient tensor in geographic coords, with
          ``strain[i, j] = d(u_j)/d(x_i)``.

    Raises:
        ValueError: If any Z > 0, or if a singular point is encountered and
            ``allow_singular`` is False.
    """
    scalar_input = np.ndim(X) == 0 and np.ndim(Y) == 0 and np.ndim(Z) == 0

    # Compute alpha: (lambda + G) / (lambda + 2*G)
    lam = 2 * G * nu / (1 - 2 * nu)
    alpha = (lam + G) / (lam + 2 * G)

    # Convert angles to radians
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)
    cs = np.cos(strike_rad)
    ss = np.sin(strike_rad)
    cd = np.cos(dip_rad)
    sd = np.sin(dip_rad)

    # Transform from geographic centroid-relative coords to DC3D internal
    # coords, following the same proven transform as okada85.setup_args():
    #   - Offset observation point for the dip-width shift
    #   - Rotate from geographic (E,N) to fault-aligned (along-strike, perp)
    #   - Compute top-edge depth from centroid depth
    X = xp.atleast_1d(xp.asarray(X))
    Y = xp.atleast_1d(xp.asarray(Y))
    d = depth + sd * width / 2  # top-edge depth

    ec = X + cs * cd * width / 2
    nc = Y - ss * cd * width / 2
    x_dc3d = cs * nc + ss * ec + length / 2
    y_dc3d = ss * nc - cs * ec + cd * width

    displacement, strain, iret = DC3D(
        alpha, x_dc3d, y_dc3d, Z, d, dip,
        F0, length, F0, width, strike_slip, dip_slip, opening,
    )

    iret_np = backend.to_numpy(iret)
    if np.any(iret_np == 2):
        bad = int(np.argmax(iret_np == 2))
        z_np = np.atleast_1d(backend.to_numpy(xp.asarray(Z)))
        raise ValueError(
            f"Invalid input: z-coordinate is positive "
            f"(Z: {z_np[min(bad, z_np.size - 1)]}). "
            "Okada92 expects Z <= 0 (at or below the free surface)."
        )
    if np.any(iret_np == 1):
        n_sing = int(np.sum(iret_np == 1))
        if not allow_singular:
            bad = int(np.argmax(iret_np.ravel() == 1))
            x_np = backend.to_numpy(X).ravel()
            y_np = backend.to_numpy(Y).ravel()
            z_np = np.broadcast_to(
                np.atleast_1d(backend.to_numpy(xp.asarray(Z))),
                backend.to_numpy(X).shape,
            ).ravel()
            bad = min(bad, x_np.size - 1)
            raise ValueError(
                f"Singular result encountered at point "
                f"(X: {x_np[bad]}, Y: {y_np[bad]}, Z: {z_np[bad]}). "
                "Set input parameter allow_singular=True to return NaN instead."
            )
        warnings.warn(
            f"Singular result at {n_sing} point(s). Outputs set to NaN."
        )

    disp_geo, strain_geo = _rotate_to_geographic(displacement, strain, ss, cs)

    if scalar_input:
        return disp_geo[0].reshape(3, 1), strain_geo[0]
    return disp_geo, strain_geo


def _rotate_to_geographic(displacement, strain, ss, cs):
    """Rotate DC3D outputs from fault-aligned to geographic coordinates.

    Handles scalar strike (``ss``/``cs`` 0-d) as well as batched strike
    arrays broadcast against leading axes of the outputs. Uses the same
    rotation as okada85: ue = sin(strike)*ux - cos(strike)*uy.
    """
    ux = displacement[..., 0]
    uy = displacement[..., 1]
    uz = displacement[..., 2]
    ue = ss * ux - cs * uy
    un = cs * ux + ss * uy
    disp_geo = xp.stack([ue, un, uz], axis=-1)

    # Rotate strain tensor from fault coords to geographic: S_geo = R S R^T
    ss_b, cs_b = xp.broadcast_arrays(xp.asarray(ss), xp.asarray(cs))
    zero = xp.zeros_like(ss_b)
    one = xp.ones_like(ss_b)
    R = xp.stack(
        [
            xp.stack([ss_b, -cs_b, zero], axis=-1),
            xp.stack([cs_b, ss_b, zero], axis=-1),
            xp.stack([zero, zero, one], axis=-1),
        ],
        axis=-2,
    )
    strain_geo = xp.einsum("...ab,...bc,...dc->...ad", R, strain, R)
    return disp_geo, strain_geo


class _C0(NamedTuple):
    """Medium and dip constants (Okada's C0 common block)."""

    alp1: object
    alp2: object
    alp3: object
    alp4: object
    alp5: object
    sd: object
    cd: object
    sdsd: object
    cdcd: object
    sdcd: object


class _C2(NamedTuple):
    """Station geometry constants (Okada's C2 common block), per point."""

    xi2: object
    et2: object
    q2: object
    r: object
    r2: object
    r3: object
    r5: object
    y: object
    d: object
    tt: object
    alx: object
    ale: object
    x11: object
    y11: object
    x32: object
    y32: object
    ey: object
    ez: object
    fy: object
    fz: object
    gy: object
    gz: object
    hy: object
    hz: object


def _dccon0(alpha, dip):
    """Compute medium constants and fault-dip constants (DCCON0).

    If cos(dip) is sufficiently small it is set to zero, as in the
    reference (### CAUTION ### block).
    """
    alp1 = (F1 - alpha) / F2
    alp2 = alpha / F2
    alp3 = (F1 - alpha) / alpha
    alp4 = F1 - alpha
    alp5 = alpha

    p18 = PI2 / 360.0
    sd = xp.sin(dip * p18)
    cd = xp.cos(dip * p18)
    small = xp.abs(cd) < EPS
    sd = xp.where(
        small, xp.where(sd > F0, F1, xp.where(sd < F0, -F1, sd)), sd
    )
    cd = xp.where(small, F0, cd)

    return _C0(alp1, alp2, alp3, alp4, alp5, sd, cd, sd * sd, cd * cd, sd * cd)


def _dccon2(xi, et, q, sd, cd, kxi, ket):
    """Compute station geometry constants for a finite source (DCCON2).

    All inputs are arrays over observation points; ``kxi``/``ket`` are the
    boolean singular-term flags (True where R+XI or R+ET vanishes). The
    scalar reference selects the singular forms with branches; here every
    branch is a ``where`` with guarded denominators, so results are
    identical on non-singular lanes and no invalid-value warnings fire.
    """
    xi = xp.where(xp.abs(xi) < EPS, F0, xi)
    et = xp.where(xp.abs(et) < EPS, F0, et)
    q = xp.where(xp.abs(q) < EPS, F0, q)

    xi2 = xi * xi
    et2 = et * et
    q2 = q * q
    r2 = xi2 + et2 + q2
    # r == 0 only exactly on a fault corner; those lanes are flagged
    # singular by the caller, so substitute a safe value to keep the
    # arithmetic finite and warning-free
    r = xp.where(r2 == F0, F1, xp.sqrt(r2))
    r2 = xp.where(r2 == F0, F1, r2)
    r3 = r * r2
    r5 = r3 * r2
    y = et * cd + q * sd
    d = et * sd - q * cd

    q_nonzero = q != F0
    q_safe = xp.where(q_nonzero, q, F1)
    tt = xp.where(q_nonzero, xp.arctan(xi * et / (q_safe * r)), F0)

    rxi = r + xi
    rxi_safe = xp.where(kxi, F1, rxi)
    rmxi_safe = xp.where(kxi, r - xi, F1)
    alx = xp.where(kxi, -xp.log(rmxi_safe), xp.log(rxi_safe))
    x11 = xp.where(kxi, F0, F1 / (r * rxi_safe))
    x32 = xp.where(kxi, F0, (r + rxi_safe) * x11 * x11 / r)

    ret = r + et
    ret_safe = xp.where(ket, F1, ret)
    rmet_safe = xp.where(ket, r - et, F1)
    ale = xp.where(ket, -xp.log(rmet_safe), xp.log(ret_safe))
    y11 = xp.where(ket, F0, F1 / (r * ret_safe))
    y32 = xp.where(ket, F0, (r + ret_safe) * y11 * y11 / r)

    ey = sd / r - y * q / r3
    ez = cd / r + d * q / r3
    fy = d / r3 + xi2 * y32 * sd
    fz = y / r3 + xi2 * y32 * cd
    gy = F2 * x11 * sd - y * q * x32
    gz = F2 * x11 * cd + d * q * x32
    hy = d * q * x32 + xi * q * y32 * sd
    hz = y * q * x32 + xi * q * y32 * cd

    return _C2(xi2, et2, q2, r, r2, r3, r5, y, d, tt, alx, ale,
               x11, y11, x32, y32, ey, ez, fy, fz, gy, gz, hy, hz)


def _ua(xi, et, q, disl1, disl2, disl3, c0, c2):
    """Displacement and strain at depth, part A (infinite-medium terms)."""
    alp1, alp2 = c0.alp1, c0.alp2
    sd, cd = c0.sd, c0.cd
    xi2, q2 = c2.xi2, c2.q2
    r, r3, y, d, tt = c2.r, c2.r3, c2.y, c2.d, c2.tt
    alx, ale = c2.alx, c2.ale
    x11, y11, y32 = c2.x11, c2.y11, c2.y32
    ey, ez, fy, fz, gy, gz, hy, hz = (c2.ey, c2.ez, c2.fy, c2.fz,
                                      c2.gy, c2.gz, c2.hy, c2.hz)

    xy = xi * y11
    qx = q * x11
    qy = q * y11

    zero = xp.zeros_like(r)
    u = [zero] * 12

    # Strike-slip contribution
    if disl1 != F0:
        du = [
            tt / F2 + alp2 * xi * qy,
            alp2 * q / r,
            alp1 * ale - alp2 * q * qy,
            -alp1 * qy - alp2 * xi2 * q * y32,
            -alp2 * xi * q / r3,
            alp1 * xy + alp2 * xi * q2 * y32,
            alp1 * xy * sd + alp2 * xi * fy + d / F2 * x11,
            alp2 * ey,
            alp1 * (cd / r + qy * sd) - alp2 * q * fy,
            alp1 * xy * cd + alp2 * xi * fz + y / F2 * x11,
            alp2 * ez,
            -alp1 * (sd / r - qy * cd) - alp2 * q * fz,
        ]
        u = [ui + disl1 / PI2 * dui for ui, dui in zip(u, du)]

    # Dip-slip contribution
    if disl2 != F0:
        du = [
            alp2 * q / r,
            tt / F2 + alp2 * et * qx,
            alp1 * alx - alp2 * q * qx,
            -alp2 * xi * q / r3,
            -qy / F2 - alp2 * et * q / r3,
            alp1 / r + alp2 * q2 / r3,
            alp2 * ey,
            alp1 * d * x11 + xy / F2 * sd + alp2 * et * gy,
            alp1 * y * x11 - alp2 * q * gy,
            alp2 * ez,
            alp1 * y * x11 + xy / F2 * cd + alp2 * et * gz,
            -alp1 * d * x11 - alp2 * q * gz,
        ]
        u = [ui + disl2 / PI2 * dui for ui, dui in zip(u, du)]

    # Tensile-fault contribution
    if disl3 != F0:
        du = [
            -alp1 * ale - alp2 * q * qy,
            -alp1 * alx - alp2 * q * qx,
            tt / F2 - alp2 * (et * qx + xi * qy),
            -alp1 * xy + alp2 * xi * q2 * y32,
            -alp1 / r + alp2 * q2 / r3,
            -alp1 * qy - alp2 * q * q2 * y32,
            -alp1 * (cd / r + qy * sd) - alp2 * q * fy,
            -alp1 * y * x11 - alp2 * q * gy,
            alp1 * (d * x11 + xy * sd) + alp2 * q * hy,
            alp1 * (sd / r - qy * cd) - alp2 * q * fz,
            alp1 * d * x11 - alp2 * q * gz,
            alp1 * (y * x11 + xy * cd) + alp2 * q * hz,
        ]
        u = [ui + disl3 / PI2 * dui for ui, dui in zip(u, du)]

    return u


def _ub(xi, et, q, disl1, disl2, disl3, c0, c2):
    """Displacement and strain at depth, part B (surface-image terms)."""
    alp3 = c0.alp3
    sd, cd, sdsd, cdcd, sdcd = c0.sd, c0.cd, c0.sdsd, c0.cdcd, c0.sdcd
    xi2, q2 = c2.xi2, c2.q2
    r, r3, y, d, tt = c2.r, c2.r3, c2.y, c2.d, c2.tt
    ale = c2.ale
    x11, y11, y32 = c2.x11, c2.y11, c2.y32
    ey, ez, fy, fz, gy, gz, hy, hz = (c2.ey, c2.ez, c2.fy, c2.fz,
                                      c2.gy, c2.gz, c2.hy, c2.hz)

    rd = r + d
    d11 = F1 / (r * rd)
    aj2 = xi * y / rd * d11
    aj5 = -(d + y * y / rd) * d11

    # The reference branches on cos(dip) == 0 (vertical fault) and on
    # xi == 0; both are expressed as where-selections with guarded
    # denominators so the kernel stays vectorized and trace-safe
    cd_nonzero = cd != F0
    cd_safe = xp.where(cd_nonzero, cd, F1)
    cdcd_safe = xp.where(cd_nonzero, cdcd, F1)
    xi_nonzero = xi != F0
    xi_safe = xp.where(xi_nonzero, xi, F1)

    x = xp.sqrt(xi2 + q2)
    ai4_gen = xp.where(
        xi_nonzero,
        F1 / cdcd_safe * (xi / rd * sdcd + F2 * xp.arctan(
            (et * (x + q * cd) + x * (r + x) * sd) / (xi_safe * (r + x) * cd_safe)
        )),
        F0,
    )
    ai3_gen = (y * cd / rd - ale + sd * xp.log(rd)) / cdcd_safe
    ak1_gen = xi * (d11 - y11 * sd) / cd_safe
    ak3_gen = (q * y11 - y * d11) / cd_safe
    aj3_gen = (ak1_gen - aj2 * sd) / cd_safe
    aj6_gen = (ak3_gen - aj5 * sd) / cd_safe

    rd2 = rd * rd
    ai3_vert = (et / rd + y * q / rd2 - ale) / F2
    ai4_vert = xi * y / rd2 / F2
    ak1_vert = xi * q / rd * d11
    ak3_vert = sd / rd * (xi2 * d11 - F1)
    aj3_vert = -xi / rd2 * (q2 * d11 - F1 / F2)
    aj6_vert = -y / rd2 * (xi2 * d11 - F1 / F2)

    ai3 = xp.where(cd_nonzero, ai3_gen, ai3_vert)
    ai4 = xp.where(cd_nonzero, ai4_gen, ai4_vert)
    ak1 = xp.where(cd_nonzero, ak1_gen, ak1_vert)
    ak3 = xp.where(cd_nonzero, ak3_gen, ak3_vert)
    aj3 = xp.where(cd_nonzero, aj3_gen, aj3_vert)
    aj6 = xp.where(cd_nonzero, aj6_gen, aj6_vert)

    xy = xi * y11
    ai1 = -xi / rd * cd - ai4 * sd
    ai2 = xp.log(rd) + ai3 * sd
    ak2 = F1 / r + ak3 * sd
    ak4 = xy * cd - ak1 * sd
    aj1 = aj5 * cd - aj6 * sd
    aj4 = -xy - aj2 * cd + aj3 * sd

    qx = q * x11
    qy = q * y11

    zero = xp.zeros_like(r)
    u = [zero] * 12

    # Strike-slip contribution
    if disl1 != F0:
        du = [
            -xi * qy - tt - alp3 * ai1 * sd,
            -q / r + alp3 * y / rd * sd,
            q * qy - alp3 * ai2 * sd,
            xi2 * q * y32 - alp3 * aj1 * sd,
            xi * q / r3 - alp3 * aj2 * sd,
            -xi * q2 * y32 - alp3 * aj3 * sd,
            -xi * fy - d * x11 + alp3 * (xy + aj4) * sd,
            -ey + alp3 * (F1 / r + aj5) * sd,
            q * fy - alp3 * (qy - aj6) * sd,
            -xi * fz - y * x11 + alp3 * ak1 * sd,
            -ez + alp3 * y * d11 * sd,
            q * fz + alp3 * ak2 * sd,
        ]
        u = [ui + disl1 / PI2 * dui for ui, dui in zip(u, du)]

    # Dip-slip contribution
    if disl2 != F0:
        du = [
            -q / r + alp3 * ai3 * sdcd,
            -et * qx - tt - alp3 * xi / rd * sdcd,
            q * qx + alp3 * ai4 * sdcd,
            xi * q / r3 + alp3 * aj4 * sdcd,
            et * q / r3 + qy + alp3 * aj5 * sdcd,
            -q2 / r3 + alp3 * aj6 * sdcd,
            -ey + alp3 * aj1 * sdcd,
            -et * gy - xy * sd + alp3 * aj2 * sdcd,
            q * gy + alp3 * aj3 * sdcd,
            -ez - alp3 * ak3 * sdcd,
            -et * gz - xy * cd - alp3 * xi * d11 * sdcd,
            q * gz - alp3 * ak4 * sdcd,
        ]
        u = [ui + disl2 / PI2 * dui for ui, dui in zip(u, du)]

    # Tensile-fault contribution
    if disl3 != F0:
        du = [
            q * qy - alp3 * ai3 * sdsd,
            q * qx + alp3 * xi / rd * sdsd,
            et * qx + xi * qy - tt - alp3 * ai4 * sdsd,
            -xi * q2 * y32 - alp3 * aj4 * sdsd,
            -q2 / r3 - alp3 * aj5 * sdsd,
            q * q2 * y32 - alp3 * aj6 * sdsd,
            q * fy - alp3 * aj1 * sdsd,
            q * gy - alp3 * aj2 * sdsd,
            -q * hy - alp3 * aj3 * sdsd,
            q * fz + alp3 * ak3 * sdsd,
            q * gz + alp3 * xi * d11 * sdsd,
            -q * hz + alp3 * ak4 * sdsd,
        ]
        u = [ui + disl3 / PI2 * dui for ui, dui in zip(u, du)]

    return u


def _uc(xi, et, q, z, disl1, disl2, disl3, c0, c2):
    """Displacement and strain at depth, part C (depth-multiplied terms)."""
    alp4, alp5 = c0.alp4, c0.alp5
    sd, cd, sdsd, cdcd, sdcd = c0.sd, c0.cd, c0.sdsd, c0.cdcd, c0.sdcd
    xi2, et2, q2 = c2.xi2, c2.et2, c2.q2
    r, r2, r3, r5, y, d = c2.r, c2.r2, c2.r3, c2.r5, c2.y, c2.d
    x11, y11, x32, y32 = c2.x11, c2.y11, c2.x32, c2.y32

    c = d + z
    x53 = (8.0 * r2 + 9.0 * r * xi + F3 * xi2) * x11 ** 3 / r2
    y53 = (8.0 * r2 + 9.0 * r * et + F3 * et2) * y11 ** 3 / r2
    h = q * cd - z
    z32 = sd / r3 - h * y32
    z53 = F3 * sd / r5 - h * y53
    y0 = y11 - xi2 * y32
    z0 = z32 - xi2 * z53
    ppy = cd / r3 + q * y32 * sd
    ppz = sd / r3 - q * y32 * cd
    qq = z * y32 + z32 + z0
    qqy = F3 * c * d / r5 - qq * sd
    qqz = F3 * c * y / r5 - qq * cd + q * y32
    xy = xi * y11
    qr = F3 * q / r5
    cqx = c * q * x53
    cdr = (c + d) / r3
    yy0 = y / r3 - y0 * cd

    zero = xp.zeros_like(r)
    u = [zero] * 12

    # Strike-slip contribution
    if disl1 != F0:
        du = [
            alp4 * xy * cd - alp5 * xi * q * z32,
            alp4 * (cd / r + F2 * q * y11 * sd) - alp5 * c * q / r3,
            alp4 * q * y11 * cd - alp5 * (c * et / r3 - z * y11 + xi2 * z32),
            alp4 * y0 * cd - alp5 * q * z0,
            -alp4 * xi * (cd / r3 + F2 * q * y32 * sd) + alp5 * c * xi * qr,
            -alp4 * xi * q * y32 * cd + alp5 * xi * (F3 * c * et / r5 - qq),
            -alp4 * xi * ppy * cd - alp5 * xi * qqy,
            alp4 * F2 * (d / r3 - y0 * sd) * sd - y / r3 * cd
            - alp5 * (cdr * sd - et / r3 - c * y * qr),
            -alp4 * q / r3 + yy0 * sd
            + alp5 * (cdr * cd + c * d * qr - (y0 * cd + q * z0) * sd),
            alp4 * xi * ppz * cd - alp5 * xi * qqz,
            alp4 * F2 * (y / r3 - y0 * cd) * sd + d / r3 * cd
            - alp5 * (cdr * cd + c * d * qr),
            yy0 * cd - alp5 * (cdr * sd - c * y * qr - y0 * sdsd + q * z0 * cd),
        ]
        u = [ui + disl1 / PI2 * dui for ui, dui in zip(u, du)]

    # Dip-slip contribution
    if disl2 != F0:
        du = [
            alp4 * cd / r - q * y11 * sd - alp5 * c * q / r3,
            alp4 * y * x11 - alp5 * c * et * q * x32,
            -d * x11 - xy * sd - alp5 * c * (x11 - q2 * x32),
            -alp4 * xi / r3 * cd + alp5 * c * xi * qr + xi * q * y32 * sd,
            -alp4 * y / r3 + alp5 * c * et * qr,
            d / r3 - y0 * sd + alp5 * c / r3 * (F1 - F3 * q2 / r2),
            -alp4 * et / r3 + y0 * sdsd - alp5 * (cdr * sd - c * y * qr),
            alp4 * (x11 - y * y * x32)
            - alp5 * c * ((d + F2 * q * cd) * x32 - y * et * q * x53),
            xi * ppy * sd + y * d * x32
            + alp5 * c * ((y + F2 * q * sd) * x32 - y * q2 * x53),
            -q / r3 + y0 * sdcd - alp5 * (cdr * cd + c * d * qr),
            alp4 * y * d * x32
            - alp5 * c * ((y - F2 * q * sd) * x32 + d * et * q * x53),
            -xi * ppz * sd + x11 - d * d * x32
            - alp5 * c * ((d - F2 * q * cd) * x32 - d * q2 * x53),
        ]
        u = [ui + disl2 / PI2 * dui for ui, dui in zip(u, du)]

    # Tensile-fault contribution
    if disl3 != F0:
        du = [
            -alp4 * (sd / r + q * y11 * cd) - alp5 * (z * y11 - q2 * z32),
            alp4 * F2 * xy * sd + d * x11 - alp5 * c * (x11 - q2 * x32),
            alp4 * (y * x11 + xy * cd) + alp5 * q * (c * et * x32 + xi * z32),
            alp4 * xi / r3 * sd + xi * q * y32 * cd
            + alp5 * xi * (F3 * c * et / r5 - F2 * z32 - z0),
            alp4 * F2 * y0 * sd - d / r3 + alp5 * c / r3 * (F1 - F3 * q2 / r2),
            -alp4 * yy0 - alp5 * (c * et * qr - q * z0),
            alp4 * (q / r3 + y0 * sdcd)
            + alp5 * (z / r3 * cd + c * d * qr - q * z0 * sd),
            -alp4 * F2 * xi * ppy * sd - y * d * x32
            + alp5 * c * ((y + F2 * q * sd) * x32 - y * q2 * x53),
            -alp4 * (xi * ppy * cd - x11 + y * y * x32)
            + alp5 * (c * ((d + F2 * q * cd) * x32 - y * et * q * x53) + xi * qqy),
            -et / r3 + y0 * cdcd
            - alp5 * (z / r3 * sd - c * y * qr - y0 * sdsd + q * z0 * cd),
            alp4 * F2 * xi * ppz * sd - x11 + d * d * x32
            - alp5 * c * ((d - F2 * q * cd) * x32 - d * q2 * x53),
            alp4 * (xi * ppz * cd + y * d * x32)
            + alp5 * (c * ((y - F2 * q * sd) * x32 + d * et * q * x53) + xi * qqz),
        ]
        u = [ui + disl3 / PI2 * dui for ui, dui in zip(u, du)]

    return u


def _edge_singular(q, xi0, xi1, et0, et1):
    """Flag observation points that lie exactly on a fault edge."""
    return (q == F0) & (
        ((xi0 * xi1 <= F0) & (et0 * et1 == F0))
        | ((et0 * et1 <= F0) & (xi0 * xi1 == F0))
    )


def _corner_flags(xi0, xi1, et0, et1, q):
    """Compute the KXI/KET singular-term flags for the four corners.

    Follows the reference exactly: KXI[k] tests R+XI at the second xi
    corner against the k-th eta corner; KET[j] tests R+ET at the second
    eta corner against the j-th xi corner.
    """
    r12 = xp.sqrt(xi0 * xi0 + et1 * et1 + q * q)
    r21 = xp.sqrt(xi1 * xi1 + et0 * et0 + q * q)
    r22 = xp.sqrt(xi1 * xi1 + et1 * et1 + q * q)
    kxi = (
        (xi0 < F0) & (r21 + xi1 < EPS),
        (xi0 < F0) & (r22 + xi1 < EPS),
    )
    ket = (
        (et0 < F0) & (r12 + et1 < EPS),
        (et0 < F0) & (r22 + et1 < EPS),
    )
    return kxi, ket


def DC3D(ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3):
    """
    ********************************************************************
    *****                                                          *****
    *****    DISPLACEMENT AND STRAIN AT DEPTH                      *****
    *****    DUE TO BURIED FINITE FAULT IN A SEMIINFINITE MEDIUM   *****
    *****              CODED BY  Y.OKADA ... SEP.1991              *****
    *****              REVISED ... NOV.1991, APR.1992, MAY.1993,   *****
    *****                          JUL.1993, MAY.2002              *****
    ********************************************************************
    *****    PYTHON VERSION BY E. LINDSEY, SEP. 2024               *****
    *****    VECTORIZED OVER OBSERVATION POINTS, JUL. 2026         *****
    ********************************************************************

    INPUT:
      ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3

      ALPHA : MEDIUM CONSTANT  (LAMBDA+MYU)/(LAMBDA+2*MYU)
      X,Y,Z : COORDINATE OF OBSERVING POINT (scalars or arrays)
      DEPTH : DEPTH OF REFERENCE POINT
      DIP   : DIP-ANGLE (DEGREE)
      AL1,AL2   : FAULT LENGTH RANGE
      AW1,AW2   : FAULT WIDTH RANGE
      DISL1-DISL3 : STRIKE-, DIP-, TENSILE-DISLOCATIONS

    OUTPUT:
      displacement: (UX, UY, UZ), shape (3, 1) for scalar input or (n, 3)
        for array input (unit of DISL)
      strain: displacement gradients, shape (3, 3) for scalar input or
        (n, 3, 3) for array input; row i holds the derivatives of
        (UX, UY, UZ) with respect to coordinate i
      IRET: return code (0 normal, 1 singular, 2 positive Z); an int for
        scalar input, an int array for array input
    """
    scalar_input = np.ndim(X) == 0 and np.ndim(Y) == 0 and np.ndim(Z) == 0
    x, y, z = xp.broadcast_arrays(
        xp.atleast_1d(xp.asarray(X)),
        xp.atleast_1d(xp.asarray(Y)),
        xp.atleast_1d(xp.asarray(Z)),
    )

    pos_z = z > F0

    c0 = _dccon0(ALPHA, DIP)
    sd, cd = c0.sd, c0.cd

    xi0 = x - AL1
    xi1 = x - AL2
    xi0 = xp.where(xp.abs(xi0) < EPS, F0, xi0)
    xi1 = xp.where(xp.abs(xi1) < EPS, F0, xi1)
    xis = (xi0, xi1)

    u = [xp.zeros_like(x) for _ in range(12)]
    singular = xp.zeros(x.shape, dtype=bool)

    # ================================================
    # =====  Real-source contribution            =====
    # ================================================
    d = DEPTH + z
    p = y * cd + d * sd
    q = y * sd - d * cd
    et0 = p - AW1
    et1 = p - AW2
    et0 = xp.where(xp.abs(et0) < EPS, F0, et0)
    et1 = xp.where(xp.abs(et1) < EPS, F0, et1)
    q = xp.where(xp.abs(q) < EPS, F0, q)
    ets = (et0, et1)

    singular = singular | _edge_singular(q, xi0, xi1, et0, et1)
    kxi, ket = _corner_flags(xi0, xi1, et0, et1, q)

    for k in range(2):
        for j in range(2):
            c2 = _dccon2(xis[j], ets[k], q, sd, cd, kxi[k], ket[j])
            dua = _ua(xis[j], ets[k], q, DISL1, DISL2, DISL3, c0, c2)

            du = [None] * 12
            for i in range(0, 12, 3):
                du[i] = -dua[i]
                du[i + 1] = -dua[i + 1] * cd + dua[i + 2] * sd
                du[i + 2] = -dua[i + 1] * sd - dua[i + 2] * cd
                if i >= 9:
                    du[i] = -du[i]
                    du[i + 1] = -du[i + 1]
                    du[i + 2] = -du[i + 2]

            sign = F1 if (j + k) != 1 else -F1
            u = [ui + sign * dui for ui, dui in zip(u, du)]

    # ================================================
    # =====  Image-source contribution           =====
    # ================================================
    d = DEPTH - z
    p = y * cd + d * sd
    q = y * sd - d * cd
    et0 = p - AW1
    et1 = p - AW2
    et0 = xp.where(xp.abs(et0) < EPS, F0, et0)
    et1 = xp.where(xp.abs(et1) < EPS, F0, et1)
    q = xp.where(xp.abs(q) < EPS, F0, q)
    ets = (et0, et1)

    singular = singular | _edge_singular(q, xi0, xi1, et0, et1)
    kxi, ket = _corner_flags(xi0, xi1, et0, et1, q)

    for k in range(2):
        for j in range(2):
            c2 = _dccon2(xis[j], ets[k], q, sd, cd, kxi[k], ket[j])
            dua = _ua(xis[j], ets[k], q, DISL1, DISL2, DISL3, c0, c2)
            dub = _ub(xis[j], ets[k], q, DISL1, DISL2, DISL3, c0, c2)
            duc = _uc(xis[j], ets[k], q, z, DISL1, DISL2, DISL3, c0, c2)

            du = [None] * 12
            for i in range(0, 12, 3):
                du[i] = dua[i] + dub[i] + z * duc[i]
                du[i + 1] = ((dua[i + 1] + dub[i + 1] + z * duc[i + 1]) * cd
                             - (dua[i + 2] + dub[i + 2] + z * duc[i + 2]) * sd)
                du[i + 2] = ((dua[i + 1] + dub[i + 1] - z * duc[i + 1]) * sd
                             + (dua[i + 2] + dub[i + 2] - z * duc[i + 2]) * cd)
                if i >= 9:
                    du[9] = du[9] + duc[0]
                    du[10] = du[10] + duc[1] * cd - duc[2] * sd
                    du[11] = du[11] - duc[1] * sd - duc[2] * cd

            sign = F1 if (j + k) != 1 else -F1
            u = [ui + sign * dui for ui, dui in zip(u, du)]

    bad = singular | pos_z
    u = [xp.where(bad, xp.nan, ui) for ui in u]
    iret = xp.where(pos_z, 2, xp.where(singular, 1, 0))

    displacement = xp.stack(u[0:3], axis=-1)
    strain = xp.stack(
        [
            xp.stack(u[3:6], axis=-1),
            xp.stack(u[6:9], axis=-1),
            xp.stack(u[9:12], axis=-1),
        ],
        axis=-2,
    )

    if scalar_input:
        return displacement[0].reshape(3, 1), strain[0], int(iret[0])
    return displacement, strain, iret
