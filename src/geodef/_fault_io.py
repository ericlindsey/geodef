"""Fault file-format I/O: center/topleft tables, unicycle seg/ned, GMT.

Private sibling of :mod:`geodef.fault` (roadmap 3.2c). ``Fault.load``,
``Fault.save``, and ``Fault.to_gmt`` dispatch to the format readers and
writers here; the geometry itself stays in the ``Fault`` class. Imports
of :class:`Fault` are deferred to call time so ``fault`` can import this
module at module level without a cycle.
"""

from typing import TYPE_CHECKING

import numpy as np

from geodef import transforms

if TYPE_CHECKING:
    from geodef.fault import Fault


def load_center(filedata: np.ndarray) -> "Fault":
    """Load patches defined by center coordinates.

    Expected columns: [id, dipid, strikeid, lon, lat, depth, L, W, strike, dip].
    """
    from geodef.fault import Fault

    lon_c = filedata[:, 3]
    lat_c = filedata[:, 4]
    depth = filedata[:, 5]
    length = filedata[:, 6]
    width = filedata[:, 7]
    strike = filedata[:, 8]
    dip = filedata[:, 9]

    n_length = int(filedata[:, 2].max()) + 1
    n_width = int(filedata[:, 1].max()) + 1
    n = len(lat_c)
    grid_shape = (n_length, n_width) if n_length * n_width == n else None

    return Fault(
        lat_c,
        lon_c,
        depth,
        strike,
        dip,
        length,
        width,
        grid_shape=grid_shape,
        engine="okada",
    )


def load_topleft(filedata: np.ndarray) -> "Fault":
    """Load patches defined by top-left corner.

    Expected columns: [id, dipid, strikeid, lon, lat, depth, L, W, strike, dip].
    """
    from geodef.fault import Fault

    lon = filedata[:, 3]
    lat = filedata[:, 4]
    depth = filedata[:, 5]
    length = filedata[:, 6]
    width = filedata[:, 7]
    strike = filedata[:, 8]
    dip = filedata[:, 9]

    sin_str = np.sin(np.radians(strike))
    cos_str = np.cos(np.radians(strike))
    sin_dip = np.sin(np.radians(dip))
    cos_dip = np.cos(np.radians(dip))

    e_offset = (length / 2) * sin_str + (width / 2) * cos_dip * cos_str
    n_offset = (length / 2) * cos_str - (width / 2) * cos_dip * sin_str
    u_offset = (width / 2) * sin_dip

    lat_c, lon_c, _ = transforms.translate_flat(lat, lon, 0.0, e_offset, n_offset, 0.0)
    depth_c = depth + u_offset

    n_length = int(filedata[:, 2].max()) + 1
    n_width = int(filedata[:, 1].max()) + 1
    n = len(lat_c)
    grid_shape = (n_length, n_width) if n_length * n_width == n else None

    return Fault(
        lat_c,
        lon_c,
        depth_c,
        strike,
        dip,
        length,
        width,
        grid_shape=grid_shape,
        engine="okada",
    )


def load_seg(fname: str, ref_lat: float, ref_lon: float) -> "Fault":
    """Load patches from a unicycle ``.seg`` file.

    Each line in the file defines a fault segment that is subdivided
    into patches using geometric growth factors. The ``.seg`` format
    uses local Cartesian coordinates (North, East, depth in meters).

    Expected columns (14-column format):
        n, Vpl, x1, x2, x3, Length, Width, Strike, Dip, Rake, L0, W0, qL, qW

    Or (13-column format, no Vpl):
        n, x1, x2, x3, Length, Width, Strike, Dip, Rake, L0, W0, qL, qW

    Args:
        fname: Path to the ``.seg`` file.
        ref_lat: Reference latitude for geographic placement.
        ref_lon: Reference longitude for geographic placement.
    """
    from geodef.fault import Fault

    # Read file, skipping comment lines
    filedata = np.loadtxt(fname, comments="#", ndmin=2)
    ncols = filedata.shape[1]

    if ncols == 14:
        # With Vpl column
        x1 = filedata[:, 2]  # North position
        x2 = filedata[:, 3]  # East position
        x3 = filedata[:, 4]  # Depth (positive down)
        seg_L = filedata[:, 5]  # Total length
        seg_W = filedata[:, 6]  # Total width
        strike = filedata[:, 7]
        dip = filedata[:, 8]
        # rake = filedata[:, 9]   # stored but not used for geometry
        L0 = filedata[:, 10]  # Initial patch length
        W0 = filedata[:, 11]  # Initial patch width
        qL = filedata[:, 12]  # Length growth factor
        qW = filedata[:, 13]  # Width growth factor
    elif ncols == 13:
        # Without Vpl column
        x1 = filedata[:, 1]
        x2 = filedata[:, 2]
        x3 = filedata[:, 3]
        seg_L = filedata[:, 4]
        seg_W = filedata[:, 5]
        strike = filedata[:, 6]
        dip = filedata[:, 7]
        # rake = filedata[:, 8]
        L0 = filedata[:, 9]
        W0 = filedata[:, 10]
        qL = filedata[:, 11]
        qW = filedata[:, 12]
    else:
        raise ValueError(f"Seg file has {ncols} columns; expected 13 or 14")

    # Process each segment and collect all patches
    all_patches = []
    for k in range(len(x1)):
        origin = np.array([x1[k], x2[k], x3[k]])  # North, East, Depth
        patches = _seg_to_patches(
            origin,
            seg_L[k],
            seg_W[k],
            strike[k],
            dip[k],
            L0[k],
            W0[k],
            qL[k],
            qW[k],
        )
        all_patches.append(patches)

    patches = np.vstack(all_patches)
    # patches columns: [north, east, depth, length, width, strike, dip]
    # These are upper-left corner positions in local Cartesian

    p_north = patches[:, 0]
    p_east = patches[:, 1]
    p_depth = patches[:, 2]
    p_length = patches[:, 3]
    p_width = patches[:, 4]
    p_strike = patches[:, 5]
    p_dip = patches[:, 6]

    # Convert upper-left corner to centroid
    sin_str = np.sin(np.radians(p_strike))
    cos_str = np.cos(np.radians(p_strike))
    sin_dip = np.sin(np.radians(p_dip))
    cos_dip = np.cos(np.radians(p_dip))

    # Strike vector: [cos(strike), sin(strike), 0] in (North, East, Down)
    # Dip vector: [-cos(dip)*sin(strike), cos(dip)*cos(strike), sin(dip)]
    # Center = corner + L/2 * strike_vec + W/2 * dip_vec
    # (following unicycle convention from flt2flt.m)
    c_north = (
        p_north + (p_length / 2) * cos_str + (p_width / 2) * (-cos_dip * sin_str)
    )
    c_east = p_east + (p_length / 2) * sin_str + (p_width / 2) * (cos_dip * cos_str)
    c_depth = p_depth + (p_width / 2) * sin_dip

    # Convert local Cartesian (East, North) to geographic
    lat_c, lon_c, _ = transforms.translate_flat(
        ref_lat,
        ref_lon,
        0.0,
        c_east,
        c_north,
        0.0,
    )
    depth_c = c_depth

    # Detect if this is a uniform grid (all patches same size, qL=qW=1)
    grid_shape = None
    if len(x1) == 1 and np.allclose(qL[0], 1.0) and np.allclose(qW[0], 1.0):
        n_width = round(seg_W[0] / W0[0])
        n_length = round(seg_L[0] / L0[0])
        if n_length * n_width == len(lat_c):
            grid_shape = (n_length, n_width)

    return Fault(
        lat_c,
        lon_c,
        depth_c,
        p_strike,
        p_dip,
        p_length,
        p_width,
        grid_shape=grid_shape,
        engine="okada",
    )


def save_center(fault: "Fault", fname: str) -> None:
    """Save in center-defined format."""
    assert fault._length is not None and fault._width is not None
    if fault._grid_shape is not None:
        nL, _ = fault._grid_shape
        strike_ids = np.arange(fault.n_patches) % nL
        dip_ids = np.arange(fault.n_patches) // nL
    else:
        strike_ids = np.zeros(fault.n_patches, dtype=int)
        dip_ids = np.arange(fault.n_patches)

    outdata = np.column_stack(
        (
            np.arange(fault.n_patches),
            dip_ids,
            strike_ids,
            fault._lon,
            fault._lat,
            fault._depth,
            fault._length,
            fault._width,
            fault.strike,
            fault.dip,
        )
    )
    np.savetxt(fname, outdata, fmt="%10.5f")


def save_seg(
    fault: "Fault",
    fname: str,
    ref_lat: float,
    ref_lon: float,
    vpl: float,
    rake: float,
) -> None:
    """Save as a unicycle ``.seg`` file.

    Writes one segment line that covers all patches. For faults with
    uniform patch sizes (qL=qW=1), this round-trips exactly. For
    non-uniform faults, L0/W0 are taken from the smallest patch.
    """
    assert fault._length is not None and fault._width is not None
    # Convert geographic centers back to local Cartesian
    alt = np.zeros(fault.n_patches)
    east, north, _ = transforms.geod2enu(
        fault._lat,
        fault._lon,
        alt,
        ref_lat,
        ref_lon,
        0.0,
    )

    # Compute upper-left corner of each patch
    sin_str = np.sin(np.radians(fault.strike))
    cos_str = np.cos(np.radians(fault.strike))
    sin_dip = np.sin(np.radians(fault.dip))
    cos_dip = np.cos(np.radians(fault.dip))

    corner_north = (
        north
        - (fault._length / 2) * cos_str
        - (fault._width / 2) * (-cos_dip * sin_str)
    )
    corner_east = (
        east
        - (fault._length / 2) * sin_str
        - (fault._width / 2) * (cos_dip * cos_str)
    )
    corner_depth = fault._depth - (fault._width / 2) * sin_dip

    # Find the overall segment bounding box
    # x1 (North), x2 (East), x3 (Depth) of the segment origin
    # = upper-left corner of the shallowest, most-negative-strike patch
    x1 = float(np.min(corner_north))
    x2 = float(np.min(corner_east))
    x3 = float(np.min(corner_depth))

    strike_val = float(fault.strike[0])
    dip_val = float(fault.dip[0])

    # Use patch properties to determine total extent
    # Sum unique lengths along strike and widths along dip
    total_L = (
        float(np.sum(fault._length[:1]))
        if fault._grid_shape is None
        else float(fault._length[0] * fault._grid_shape[0])
    )
    total_W = (
        float(np.sum(fault._width[:1]))
        if fault._grid_shape is None
        else float(fault._width[0] * fault._grid_shape[1])
    )

    # If we don't have grid_shape, estimate from the patches
    if fault._grid_shape is None:
        # Sum all unique widths (down-dip) and max strike extent
        total_L = float(np.max(fault._length) * len(fault._length))
        total_W = float(np.sum(np.unique(fault._width)))

    L0 = float(np.min(fault._length))
    W0 = float(np.min(fault._width))

    # Detect growth factors
    if fault._grid_shape is not None:
        qL = 1.0
        qW = 1.0
    else:
        unique_widths = np.unique(fault._width)
        if len(unique_widths) > 1:
            qW = float(unique_widths[1] / unique_widths[0])
        else:
            qW = 1.0
        unique_lengths = np.unique(fault._length)
        if len(unique_lengths) > 1:
            qL = float(unique_lengths[1] / unique_lengths[0])
        else:
            qL = 1.0

    with open(fname, "w") as f:
        f.write("# Unicycle .seg file generated by geodef\n")
        f.write(
            "# n  Vpl  x1  x2  x3  Length  Width  Strike  Dip  Rake  "
            "L0  W0  qL  qW\n"
        )
        f.write(
            f"1 {vpl:.9f} {x1:.9f} {x2:.9f} {x3:.9f} "
            f"{total_L:.9f} {total_W:.9f} {strike_val:.9f} {dip_val:.9f} "
            f"{rake:.9f} {L0:.9f} {W0:.9f} {qL:.9f} {qW:.9f}\n"
        )


def save_tri_ned(fault: "Fault", fname: str) -> None:
    """Save triangular fault as a unicycle .ned + .tri file pair.

    Reconstructs the mesh node/triangle topology by deduplicating the
    stored ENU vertex positions and converting back to geographic
    coordinates.
    """
    from geodef.mesh import Mesh

    assert fault._vertices is not None
    n_tri = fault.n_patches
    verts_flat = fault._vertices.reshape(-1, 3)  # (N*3, 3) [east, north, up]

    geographic = fault._frame.to_geographic(
        east=verts_flat[:, 0],
        north=verts_flat[:, 1],
        up=verts_flat[:, 2],
    )
    lon_nodes = geographic[:, 0]
    lat_nodes = geographic[:, 1]
    depth_nodes = -geographic[:, 2]

    # Deduplicate nodes with fixed precision to merge shared vertices
    coords = np.column_stack([lon_nodes, lat_nodes, depth_nodes])
    coords_rounded = np.round(coords, 6)
    _, unique_idx, inverse = np.unique(
        coords_rounded, axis=0, return_index=True, return_inverse=True
    )

    mesh = Mesh(
        lon=lon_nodes[unique_idx],
        lat=lat_nodes[unique_idx],
        depth=depth_nodes[unique_idx],
        triangles=inverse.reshape(n_tri, 3),
        frame=fault._frame,
    )
    mesh.save(fname)


def write_gmt(fault: "Fault", fname: str, values: np.ndarray | None = None) -> None:
    """Export fault patches as a GMT multi-segment polygon file.

    Each patch becomes one closed polygon segment with a ``> -Z{value}``
    header line.  The file is compatible with GMT's ``psxy -L`` (closed
    polygons) and ``-Z`` coloring.

    Args:
        fault: The fault to export.
        fname: Output file path.
        values: Scalar value per patch (e.g. slip magnitude), shape (N,).
            Defaults to zeros if not provided.

    Raises:
        ValueError: If ``values`` is provided but has the wrong length.
    """
    n = fault.n_patches
    if values is None:
        values = np.zeros(n)
    else:
        values = np.asarray(values, dtype=float)
        if values.shape != (n,):
            raise ValueError(f"values must have shape ({n},), got {values.shape}")

    if fault._engine == "okada":
        # vertices_2d: (N, 4, 2) as [lon, lat]
        verts = fault.vertices_2d  # (N, 4, 2)
    else:
        # Triangular: convert ENU to geographic lon/lat
        verts_enu = fault._vertices  # (N, 3, 3)
        assert verts_enu is not None
        verts_flat = verts_enu.reshape(-1, 3)
        geographic = fault._frame.to_geographic(
            east=verts_flat[:, 0],
            north=verts_flat[:, 1],
            up=verts_flat[:, 2],
        )
        verts = np.stack(
            [
                geographic[:, 0].reshape(n, 3),
                geographic[:, 1].reshape(n, 3),
            ],
            axis=-1,
        )  # (N, 3, 2)

    with open(fname, "w") as fh:
        fh.write("# geodef GMT fault export\n")
        for i in range(n):
            fh.write(f"> -Z{values[i]:.6f}\n")
            for corner in range(verts.shape[1]):
                fh.write(f"{verts[i, corner, 0]:.6f} {verts[i, corner, 1]:.6f}\n")


def _seg_to_patches(
    origin: np.ndarray,
    total_L: float,
    total_W: float,
    strike: float,
    dip: float,
    L0: float,
    W0: float,
    alpha_l: float,
    alpha_w: float,
) -> np.ndarray:
    """Subdivide a fault segment into patches with geometric growth.

    Port of unicycle's ``flt2flt.m``. Patch widths grow geometrically
    with depth (down-dip). For each dip row, patch lengths also grow
    geometrically but are distributed evenly across the strike direction.

    Args:
        origin: Upper-left corner [North, East, Depth] in meters.
        total_L: Total along-strike length in meters.
        total_W: Total down-dip width in meters.
        strike: Strike angle in degrees.
        dip: Dip angle in degrees.
        L0: Initial (smallest) patch length.
        W0: Initial (smallest) patch width.
        alpha_l: Geometric growth factor for length (1.0 = uniform).
        alpha_w: Geometric growth factor for width (1.0 = uniform).

    Returns:
        Array of shape (N, 7) with columns
        [north, east, depth, length, width, strike, dip]
        where (north, east, depth) is the upper-left corner of each patch.
    """
    # Step 1: Compute down-dip width distribution
    remaining_W = total_W
    widths = []
    k = 0
    while remaining_W > 0:
        wt = W0 * alpha_w**k
        if wt > remaining_W / 2:
            wt = remaining_W
        wt = min(wt, remaining_W)
        widths.append(wt)
        remaining_W -= wt
        k += 1

    # Step 2: Strike and dip direction unit vectors
    # Unicycle convention: strike_vec = [cos(strike), sin(strike), 0]
    # dip_vec = [-cos(dip)*sin(strike), cos(dip)*cos(strike), sin(dip)]
    str_rad = np.radians(strike)
    dip_rad = np.radians(dip)
    strike_vec = np.array([np.cos(str_rad), np.sin(str_rad), 0.0])
    dip_vec = np.array(
        [
            -np.cos(dip_rad) * np.sin(str_rad),
            np.cos(dip_rad) * np.cos(str_rad),
            np.sin(dip_rad),
        ]
    )

    # Step 3: Build patches row by row
    patches = []
    cumulative_w = 0.0
    for j, wj in enumerate(widths):
        # Patch length for this row
        lt = L0 * alpha_l**j
        n_along = int(np.ceil(total_L / lt))
        lt = total_L / n_along  # distribute evenly

        for i in range(n_along):
            corner = origin + i * lt * strike_vec + cumulative_w * dip_vec
            patches.append(
                [
                    corner[0],
                    corner[1],
                    corner[2],
                    lt,
                    wj,
                    strike,
                    dip,
                ]
            )
        cumulative_w += wj

    return np.array(patches)
