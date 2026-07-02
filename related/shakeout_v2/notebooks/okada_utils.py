import numpy as np
import okada85

def fault_outline(depth_m, dip_deg, length_m, width_m, strike_deg, centroid_E_m, centroid_N_m, return_depths=False):
    strike = np.deg2rad(strike_deg)
    dip = np.deg2rad(dip_deg)
    u_strike = np.array([np.sin(strike), np.cos(strike)])
    u_dip_h = np.array([np.sin(strike + 0.5 * np.pi), np.cos(strike + 0.5 * np.pi)])

    half_L = 0.5 * length_m * u_strike
    half_W_h = 0.5 * width_m * np.cos(dip) * u_dip_h

    C = np.array([centroid_E_m, centroid_N_m])
    top_center = C - half_W_h
    bot_center = C + half_W_h

    top_left = top_center - half_L
    top_right = top_center + half_L
    bot_left = bot_center - half_L
    bot_right = bot_center + half_L

    if return_depths:
        # Depth logic from 03 notebook if necessary, else just ignore and return 2D outline
        # Notebook 04/05 don't use return_depths
        pass

    return np.vstack([top_left, top_right, bot_right, bot_left, top_left])

def build_patch_grid(e0, n0, z0, strike_deg, dip_deg, fault_L, fault_W, nL, nW):
    """
    Build patch centers and geometry.

    ASCII index sketch:
        j=0  [0,0] [1,0] [2,0] ...
        j=1  [0,1] [1,1] [2,1] ...

        i increases along strike
        j increases down dip
    """
    patch_L = fault_L / nL
    patch_W = fault_W / nW

    strike = np.deg2rad(strike_deg)
    dip = np.deg2rad(dip_deg)
    sin_str, cos_str = np.sin(strike), np.cos(strike)
    sin_dip, cos_dip = np.sin(dip), np.cos(dip)

    fault_eoffset = -0.5 * fault_L * sin_str - 0.5 * fault_W * cos_dip * cos_str
    fault_noffset = -0.5 * fault_L * cos_str + 0.5 * fault_W * cos_dip * sin_str
    fault_uoffset = -0.5 * fault_W * sin_dip

    patches = []
    for j in range(nW):
        for i in range(nL):
            e = e0 + fault_eoffset + (i + 0.5) * patch_L * sin_str + (j + 0.5) * patch_W * cos_dip * cos_str
            n = n0 + fault_noffset + (i + 0.5) * patch_L * cos_str - (j + 0.5) * patch_W * cos_dip * sin_str
            u = fault_uoffset + (j + 0.5) * patch_W * sin_dip
            depth = z0 - u

            patches.append({
                'i': i, 'j': j,
                'e': float(e), 'n': float(n), 'depth': float(depth),
                'strike': float(strike_deg), 'dip': float(dip_deg),
                'L': float(patch_L), 'W': float(patch_W),
            })
    return patches


def build_component_greens(obs_e, obs_n, patches, rake_deg=90.0, nu=0.25):
    """Return GE, GN, GU where each column is unit-slip response of one patch."""
    nobs = len(obs_e)
    npatch = len(patches)
    GE = np.zeros((nobs, npatch))
    GN = np.zeros((nobs, npatch))
    GU = np.zeros((nobs, npatch))

    for p, patch in enumerate(patches):
        e_rel = obs_e - patch['e']
        n_rel = obs_n - patch['n']
        uE, uN, uU = okada85.displacement(
            e_rel, n_rel,
            patch['depth'], patch['strike'], patch['dip'], patch['L'], patch['W'],
            rake_deg, 1.0, 0.0, nu
        )
        GE[:, p] = uE
        GN[:, p] = uN
        GU[:, p] = uU

    return GE, GN, GU


def build_laplacian_2d(nL, nW):
    """
    Build a 2D finite difference Laplacian matrix for a rectangular fault 
    with nL patches along strike and nW patches down dip.
    
    The matrix size is (nL * nW, nL * nW). It uses second-order central 
    differences for the interior and second-order forward/backward 
    differences at the boundaries.
    """
    if nL < 3 or nW < 3:
        raise ValueError("Laplacian calculation requires at least 3 patches in each dimension.")

    npatch = nL * nW
    L = np.zeros((npatch, npatch))

    for j in range(nW):
        for i in range(nL):
            k = j * nL + i

            # Along-strike direction (i)
            if 0 < i < nL - 1:
                # Central difference
                L[k, k - 1] += 1.0
                L[k, k]     -= 2.0
                L[k, k + 1] += 1.0
            elif i == 0:
                # Forward difference
                L[k, k]     += 1.0
                L[k, k + 1] -= 2.0
                L[k, k + 2] += 1.0
            elif i == nL - 1:
                # Backward difference
                L[k, k]     += 1.0
                L[k, k - 1] -= 2.0
                L[k, k - 2] += 1.0

            # Down-dip direction (j)
            if 0 < j < nW - 1:
                # Central difference
                L[k, k - nL] += 1.0
                L[k, k]      -= 2.0
                L[k, k + nL] += 1.0
            elif j == 0:
                # Forward difference
                L[k, k]        += 1.0
                L[k, k + nL]   -= 2.0
                L[k, k + 2 * nL] += 1.0
            elif j == nW - 1:
                # Backward difference
                L[k, k]        += 1.0
                L[k, k - nL]   -= 2.0
                L[k, k - 2 * nL] += 1.0

    return L

def build_laplacian_2d_simple(nL, nW):
    """
    Build a 2D finite difference Laplacian matrix for a rectangular fault 
    with nL patches along strike and nW patches down dip.
    
    The matrix size is (nL * nW, nL * nW). It correctly handles corner and edge 
    cases by adjusting the diagonal weight to equal the number of available 
    neighboring patches, ensuring the sum of each row is zero (free boundary condition).
    """
    npatch = nL * nW
    L = np.zeros((npatch, npatch))

    for j in range(nW):
        for i in range(nL):
            k = j * nL + i
            num_neighbors = 0

            # Left neighbor
            if i > 0:
                L[k, k - 1] = 1.0
                num_neighbors += 1
            # Right neighbor
            if i < nL - 1:
                L[k, k + 1] = 1.0
                num_neighbors += 1
            # Top neighbor (up dip)
            if j > 0:
                L[k, k - nL] = 1.0
                num_neighbors += 1
            # Bottom neighbor (down dip)
            if j < nW - 1:
                L[k, k + nL] = 1.0
                num_neighbors += 1
                
            L[k, k] = -float(num_neighbors)

    return L
