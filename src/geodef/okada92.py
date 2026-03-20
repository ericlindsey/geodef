import warnings

import numpy as np
import math

def okada92(X, Y, Z, depth, strike, dip, length, width,
            strike_slip, dip_slip, opening, G, nu, allow_singular=False):
    """Compute displacement and strain at depth due to a rectangular dislocation.

    Wrapper around DC3D (Okada, 1992) that accepts geographic coordinates
    (East, North, Up) relative to the fault centroid and returns results in
    the same geographic frame.

    Args:
        X: Easting of observation point relative to fault centroid.
        Y: Northing of observation point relative to fault centroid.
        Z: Observation depth (Z <= 0, with Z=0 at the free surface).
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
        - displacement: shape (3,1) array [ue, un, uz] in geographic coords.
        - strain: shape (3,3) displacement gradient tensor in geographic coords.
    """
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

    # Transform from geographic centroid-relative coords to DC3D internal coords.
    # This follows the same proven transform as okada85.setup_args():
    #   - Offset observation point for the dip-width shift
    #   - Rotate from geographic (E,N) to fault-aligned (along-strike, perp-strike)
    #   - Compute top-edge depth from centroid depth
    d = depth + sd * width / 2  # top-edge depth

    ec = X + cs * cd * width / 2
    nc = Y - ss * cd * width / 2
    x_dc3d = cs * nc + ss * ec + length / 2
    y_dc3d = ss * nc - cs * ec + cd * width

    # DC3D uses fault-length bounds relative to reference point
    al1 = 0.0
    al2 = length
    aw1 = 0.0
    aw2 = width

    # Call the core DC3D engine
    displacement, strain, iret = DC3D(
        alpha, x_dc3d, y_dc3d, Z, d, dip,
        al1, al2, aw1, aw2, strike_slip, dip_slip, opening,
    )

    if iret == 1:
        if not allow_singular:
            raise ValueError(
                f"Singular result encountered at point (X: {X}, Y: {Y}, Z: {Z}). "
                "Set input parameter allow_singular=True to return NaN instead."
            )
        else:
            displacement = np.full((3, 1), np.nan)
            strain = np.full((3, 3), np.nan)
            warnings.warn(
                f"Singular result at ({X}, {Y}, {Z}). Outputs set to NaN."
            )
            return displacement, strain

    if iret == 2:
        raise ValueError(
            f"Invalid input: z-coordinate is positive (Z: {Z}). "
            "Okada92 expects Z <= 0 (at or below the free surface)."
        )

    # Rotate displacements from fault-aligned (x=strike, y=perp) to geographic (E, N)
    # Uses the same rotation as okada85: ue = sin(strike)*ux - cos(strike)*uy
    ux = displacement[0, 0]
    uy = displacement[1, 0]
    uz = displacement[2, 0]
    displacement[0, 0] = ss * ux - cs * uy  # East
    displacement[1, 0] = cs * ux + ss * uy  # North
    displacement[2, 0] = uz                 # Up (unchanged)

    # Rotate strain tensor from fault coords to geographic: S_geo = R @ S_fault @ R^T
    # where R rotates fault-aligned axes to geographic axes.
    R = np.array([[ss, -cs, 0],
                  [cs,  ss, 0],
                  [ 0,   0, 1]])
    strain = R @ strain @ R.T

    return displacement, strain


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
    ********************************************************************
    
    INPUT:
      ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3

      ALPHA : MEDIUM CONSTANT  (LAMBDA+MYU)/(LAMBDA+2*MYU)
      X,Y,Z : COORDINATE OF OBSERVING POINT
      DEPTH : DEPTH OF REFERENCE POINT
      DIP   : DIP-ANGLE (DEGREE)
      AL1,AL2   : FAULT LENGTH RANGE
      AW1,AW2   : FAULT WIDTH RANGE
      DISL1-DISL3 : STRIKE-, DIP-, TENSILE-DISLOCATIONS

    OUTPUT:
      (UX, UY, UZ): 3x1 array for displacement (unit of DISL)
      (UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ): 3x3 array for strain derivatives (unit of DISL / Unit of X,Y,Z,Depth, AL, AW)
      IRET: return code (0 for normal, 1 for singular, 2 for positive Z)

    """

    global SD,CD
    
    # Constants
    F0, EPS = 0.0, 1e-6
    PI2 = 6.283185307179586
    
    # Initialize U and derivatives
    U = np.zeros(12)
    DUA = np.zeros(12)
    DUB = np.zeros(12)
    DUC = np.zeros(12)
    
    # Set IRET flag
    IRET = 0
    if Z > 0:
        # reject positive Z
        IRET = 2
        return np.full((3, 1), np.nan), np.full((3, 3), np.nan), IRET
    
    # Call DCCON0
    AALPHA = ALPHA
    DDIP = DIP
    DCCON0(AALPHA, DDIP)
    
    # Coordinates
    ZZ = Z
    DD1 = DISL1
    DD2 = DISL2
    DD3 = DISL3
    XI = [X - AL1, X - AL2]

    # Real source setup
    D = DEPTH + Z
    P = Y * CD + D * SD
    Q = Y * SD - D * CD
    ET = [P - AW1, P - AW2]

    # Handle small values
    XI = [F0 if abs(x) < EPS else x for x in XI]
    ET = [F0 if abs(e) < EPS else e for e in ET]

    if abs(Q) < EPS:
        Q = F0

    # Reject singular cases (on fault edge)
    if (Q == F0 and ((XI[0] * XI[1] <= F0 and ET[0] * ET[1] == F0) or 
                     (ET[0] * ET[1] <= F0 and XI[0] * XI[1] == F0))):
        IRET = 1
        return np.full((3, 1), np.nan), np.full((3, 3), np.nan), IRET

    # Initialize KXI and KET
    KXI = [0, 0]
    KET = [0, 0]

    # Compute distances
    R12 = math.sqrt(XI[0]**2 + ET[1]**2 + Q**2)
    R21 = math.sqrt(XI[1]**2 + ET[0]**2 + Q**2)
    R22 = math.sqrt(XI[1]**2 + ET[1]**2 + Q**2)
    
    if XI[0] < 0.0 and R21 + XI[1] < EPS:
        KXI[0] = 1
    if XI[0] < 0.0 and R22 + XI[1] < EPS:
        KXI[1] = 1
    if ET[0] < 0.0 and R12 + ET[1] < EPS:
        KET[0] = 1
    if ET[0] < 0.0 and R22 + ET[1] < EPS:
        KET[1] = 1

    DU=np.zeros(12)

    for K in range(2):
        for J in range(2):
            DCCON2(XI[J], ET[K], Q, SD, CD, KXI[K], KET[J])  # Call DCCON2
            UA(XI[J], ET[K], Q, DD1, DD2, DD3, DUA)                      # Call UA

            # Update DU with transformations
            for I in range(0, 12, 3):
                DU[I] = -DUA[I]
                DU[I+1] = -DUA[I+1] * CD + DUA[I+2] * SD
                DU[I+2] = -DUA[I+1] * SD - DUA[I+2] * CD
                if I >= 9:
                    DU[I] = -DU[I]
                    DU[I+1] = -DU[I+1]
                    DU[I+2] = -DU[I+2]

            for I in range(0, 12):
                if J + K != 1:
                    U[I] += DU[I]
                else:
                    U[I] -= DU[I]

    # Image source setup
    D = DEPTH - Z
    P = Y * CD + D * SD
    Q = Y * SD - D * CD
    ET = [P - AW1, P - AW2]
    
    # Handle small values
    ET = [F0 if abs(e) < EPS else e for e in ET]
    
    if abs(Q) < EPS:
        Q = F0

    # Reject singular case (on fault edge)
    if (Q == F0 and ((XI[0] * XI[1] <= F0 and ET[0] * ET[1] == F0) or 
                     (ET[0] * ET[1] <= F0 and XI[0] * XI[1] == F0))):
        IRET = 1
        return np.full((3, 1), np.nan), np.full((3, 3), np.nan), IRET

    # Initialize KXI and KET
    KXI = [0, 0]
    KET = [0, 0]
    
    # Compute distances
    R12 = math.sqrt(XI[0]**2 + ET[1]**2 + Q**2)
    R21 = math.sqrt(XI[1]**2 + ET[0]**2 + Q**2)
    R22 = math.sqrt(XI[1]**2 + ET[1]**2 + Q**2)
    
    if XI[0] < 0.0 and R21 + XI[1] < EPS:
        KXI[0] = 1
    if XI[0] < 0.0 and R22 + XI[1] < EPS:
        KXI[1] = 1
    if ET[0] < 0.0 and R12 + ET[1] < EPS:
        KET[0] = 1
    if ET[0] < 0.0 and R22 + ET[1] < EPS:
        KET[1] = 1
    
    for K in range(2):
        for J in range(2):
            DCCON2(XI[J], ET[K], Q, SD, CD, KXI[K], KET[J])  # Compute constants
            UA(XI[J], ET[K], Q, DD1, DD2, DD3, DUA)          # Call UA
            UB(XI[J], ET[K], Q, DD1, DD2, DD3, DUB)          # Call UB
            UC(XI[J], ET[K], Q, Z, DD1, DD2, DD3, DUC)       # Call UC
            
            # Combine results
            for I in range(0, 12, 3):
                DU[I] = DUA[I] + DUB[I] + Z * DUC[I]
                DU[I+1] = (DUA[I+1] + DUB[I+1] + Z * DUC[I+1]) * CD - (DUA[I+2] + DUB[I+2] + Z * DUC[I+2]) * SD
                DU[I+2] = (DUA[I+1] + DUB[I+1] - Z * DUC[I+1]) * SD + (DUA[I+2] + DUB[I+2] - Z * DUC[I+2]) * CD
                
                if I >= 9:
                    DU[9] = DU[9] + DUC[0]
                    DU[10] = DU[10] + DUC[1] * CD - DUC[2] * SD
                    DU[11] = DU[11] - DUC[1] * SD - DUC[2] * CD 
            
            for I in range(0, 12):
                if J + K != 1:
                    U[I] = U[I] + DU[I]
                else:
                    U[I] = U[I] - DU[I]
    
    # Extract displacements and strains
    UX, UY, UZ = U[0], U[1], U[2]
    UXX, UYX, UZX = U[3], U[4], U[5]
    UXY, UYY, UZY = U[6], U[7], U[8]
    UXZ, UYZ, UZZ = U[9], U[10], U[11]

    # Return as NumPy arrays
    displacement = np.array([[UX], [UY], [UZ]])
    strain = np.array([[UXX, UYX, UZX], [UXY, UYY, UZY], [UXZ, UYZ, UZZ]])

    return displacement, strain, IRET



def UA(XI, ET, Q, DISL1, DISL2, DISL3, U):
    """
    ********************************************************************
    *****    DISPLACEMENT AND STRAIN AT DEPTH (PART-A)             *****
    *****    DUE TO BURIED FINITE FAULT IN A SEMIINFINITE MEDIUM   *****
    ********************************************************************
    
    INPUT
      XI, ET, Q : STATION COORDINATES IN FAULT SYSTEM
      DISL1-DISL3 : STRIKE-, DIP-, TENSILE-DISLOCATIONS
    OUTPUT
      U(12) : DISPLACEMENT AND THEIR DERIVATIVES
    """
    global ALP1, ALP2, ALP3, ALP4, ALP5, SD, CD, SDSD, CDCD, SDCD, S2D, C2D
    global XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32
    global EY, EZ, FY, FZ, GY, GZ, HY, HZ
    
    F0, F2, PI2 = 0.0, 2.0, 6.283185307179586
    
    DU = [0.0] * 12
    
    # Initialize U to zeros
    for i in range(12):
        U[i] = F0
    
    XY = XI * Y11
    QX = Q * X11
    QY = Q * Y11
    
    # Strike-slip contribution
    if DISL1 != F0:
        DU[0]  = TT / F2 + ALP2 * XI * QY
        DU[1]  = ALP2 * Q / R
        DU[2]  = ALP1 * ALE - ALP2 * Q * QY
        DU[3]  = -ALP1 * QY - ALP2 * XI2 * Q * Y32
        DU[4]  = -ALP2 * XI * Q / R3
        DU[5]  = ALP1 * XY + ALP2 * XI * Q2 * Y32
        DU[6]  = ALP1 * XY * SD + ALP2 * XI * FY + D / F2 * X11
        DU[7]  = ALP2 * EY
        DU[8]  = ALP1 * (CD / R + QY * SD) - ALP2 * Q * FY
        DU[9]  = ALP1 * XY * CD + ALP2 * XI * FZ + Y / F2 * X11
        DU[10] = ALP2 * EZ
        DU[11] = -ALP1 * (SD / R - QY * CD) - ALP2 * Q * FZ
        
        for i in range(12):
            U[i] += DISL1 / PI2 * DU[i]

    # Dip-slip contribution
    if DISL2 != F0:
        DU[0]  = ALP2 * Q / R
        DU[1]  = TT / F2 + ALP2 * ET * QX
        DU[2]  = ALP1 * ALX - ALP2 * Q * QX
        DU[3]  = -ALP2 * XI * Q / R3
        DU[4]  = -QY / F2 - ALP2 * ET * Q / R3
        DU[5]  = ALP1 / R + ALP2 * Q2 / R3
        DU[6]  = ALP2 * EY
        DU[7]  = ALP1 * D * X11 + XY / F2 * SD + ALP2 * ET * GY
        DU[8]  = ALP1 * Y * X11 - ALP2 * Q * GY
        DU[9]  = ALP2 * EZ
        DU[10] = ALP1 * Y * X11 + XY / F2 * CD + ALP2 * ET * GZ
        DU[11] = -ALP1 * D * X11 - ALP2 * Q * GZ
        
        for i in range(12):
            U[i] += DISL2 / PI2 * DU[i]

    # Tensile-fault contribution
    if DISL3 != F0:
        DU[0]  = -ALP1 * ALE - ALP2 * Q * QY
        DU[1]  = -ALP1 * ALX - ALP2 * Q * QX
        DU[2]  = TT / F2 - ALP2 * (ET * QX + XI * QY)
        DU[3]  = -ALP1 * XY + ALP2 * XI * Q2 * Y32
        DU[4]  = -ALP1 / R + ALP2 * Q2 / R3
        DU[5]  = -ALP1 * QY - ALP2 * Q * Q2 * Y32
        DU[6]  = -ALP1 * (CD / R + QY * SD) - ALP2 * Q * FY
        DU[7]  = -ALP1 * Y * X11 - ALP2 * Q * GY
        DU[8]  = ALP1 * (D * X11 + XY * SD) + ALP2 * Q * HY
        
        for i in range(12):
            U[i] += DISL3 / PI2 * DU[i]

    return


def UB(XI, ET, Q, DISL1, DISL2, DISL3, U):
    """
    ********************************************************************
    *****    DISPLACEMENT AND STRAIN AT DEPTH (PART-B)             *****
    *****    DUE TO BURIED FINITE FAULT IN A SEMIINFINITE MEDIUM   *****
    ********************************************************************

    INPUT
      XI, ET, Q : STATION COORDINATES IN FAULT SYSTEM
      DISL1-DISL3 : STRIKE-, DIP-, TENSILE-DISLOCATIONS
    OUTPUT
      U(12) : DISPLACEMENT AND THEIR DERIVATIVES
    """
    global ALP1, ALP2, ALP3, ALP4, ALP5, SD, CD, SDSD, CDCD, SDCD, S2D, C2D
    global XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32
    global EY, EZ, FY, FZ, GY, GZ, HY, HZ

    F0, F1, F2, PI2 = 0.0, 1.0, 2.0, 6.283185307179586

    DU = [0.0] * 12

    # Initial computations
    RD = R + D
    D11 = F1 / (R * RD)
    AJ2 = XI * Y / RD * D11
    AJ5 = -(D + Y * Y / RD) * D11

    if CD != F0:
        if XI == F0:
            AI4 = F0
        else:
            X = (XI2 + Q2)**0.5
            AI4 = F1 / CDCD * (XI / RD * SDCD + F2 * math.atan((ET * (X + Q * CD) + X * (R + X) * SD) / (XI * (R + X) * CD)))
        AI3 = (Y * CD / RD - ALE + SD * math.log(RD)) / CDCD
        AK1 = XI * (D11 - Y11 * SD) / CD
        AK3 = (Q * Y11 - Y * D11) / CD
        AJ3 = (AK1 - AJ2 * SD) / CD
        AJ6 = (AK3 - AJ5 * SD) / CD
    else:
        RD2 = RD * RD
        AI3 = (ET / RD + Y * Q / RD2 - ALE) / F2
        AI4 = XI * Y / RD2 / F2
        AK1 = XI * Q / RD * D11
        AK3 = SD / RD * (XI2 * D11 - F1)
        AJ3 = -XI / RD2 * (Q2 * D11 - F1 / F2)
        AJ6 = -Y / RD2 * (XI2 * D11 - F1 / F2)

    # Further calculations
    XY = XI * Y11
    AI1 = -XI / RD * CD - AI4 * SD
    AI2 = math.log(RD) + AI3 * SD
    AK2 = F1 / R + AK3 * SD
    AK4 = XY * CD - AK1 * SD
    AJ1 = AJ5 * CD - AJ6 * SD
    AJ4 = -XY - AJ2 * CD + AJ3 * SD

    # Initialize U to zeros
    for i in range(12):
        U[i] = F0

    QX = Q * X11
    QY = Q * Y11

    # Strike-slip contribution
    if DISL1 != F0:
        DU[0] = -XI * QY - TT - ALP3 * AI1 * SD
        DU[1] = -Q / R + ALP3 * Y / RD * SD
        DU[2] = Q * QY - ALP3 * AI2 * SD
        DU[3] = XI2 * Q * Y32 - ALP3 * AJ1 * SD
        DU[4] = XI * Q / R3 - ALP3 * AJ2 * SD
        DU[5] = -XI * Q2 * Y32 - ALP3 * AJ3 * SD
        DU[6] = -XI * FY - D * X11 + ALP3 * (XY + AJ4) * SD
        DU[7] = -EY + ALP3 * (F1 / R + AJ5) * SD
        DU[8] = Q * FY - ALP3 * (QY - AJ6) * SD
        DU[9] = -XI * FZ - Y * X11 + ALP3 * AK1 * SD
        DU[10] = -EZ + ALP3 * Y * D11 * SD
        DU[11] = Q * FZ + ALP3 * AK2 * SD

        for i in range(12):
            U[i] += DISL1 / PI2 * DU[i]

    # Dip-slip contribution
    if DISL2 != F0:
        DU[0] = -Q / R + ALP3 * AI3 * SDCD
        DU[1] = -ET * QX - TT - ALP3 * XI / RD * SDCD
        DU[2] = Q * QX + ALP3 * AI4 * SDCD
        DU[3] = XI * Q / R3 + ALP3 * AJ4 * SDCD
        DU[4] = ET * Q / R3 + QY + ALP3 * AJ5 * SDCD
        DU[5] = -Q2 / R3 + ALP3 * AJ6 * SDCD
        DU[6] = -EY + ALP3 * AJ1 * SDCD
        DU[7] = -ET * GY - XY * SD + ALP3 * AJ2 * SDCD
        DU[8] = Q * GY + ALP3 * AJ3 * SDCD
        DU[9] = -EZ - ALP3 * AK3 * SDCD
        DU[10] = -ET * GZ - XY * CD - ALP3 * XI * D11 * SDCD
        DU[11] = Q * GZ - ALP3 * AK4 * SDCD

        for i in range(12):
            U[i] += DISL2 / PI2 * DU[i]

    # Tensile-fault contribution
    if DISL3 != F0:
        DU[0] = Q * QY - ALP3 * AI3 * SDSD
        DU[1] = Q * QX + ALP3 * XI / RD * SDSD
        DU[2] = ET * QX + XI * QY - TT - ALP3 * AI4 * SDSD
        DU[3] = -XI * Q2 * Y32 - ALP3 * AJ4 * SDSD
        DU[4] = -Q2 / R3 - ALP3 * AJ5 * SDSD
        DU[5] = Q * Q2 * Y32 - ALP3 * AJ6 * SDSD
        DU[6] = Q * FY - ALP3 * AJ1 * SDSD
        DU[7] = Q * GY - ALP3 * AJ2 * SDSD
        DU[8] = -Q * HY - ALP3 * AJ3 * SDSD
        DU[9] = Q * FZ + ALP3 * AK3 * SDSD

        for i in range(12):
            U[i] += DISL3 / PI2 * DU[i]

    return


def UC(XI, ET, Q, Z, DISL1, DISL2, DISL3, U):
    """
    ********************************************************************
    *****    DISPLACEMENT AND STRAIN AT DEPTH (PART-C)             *****
    *****    DUE TO BURIED FINITE FAULT IN A SEMIINFINITE MEDIUM   *****
    ********************************************************************
    
    INPUT
      XI, ET, Q, Z   : STATION COORDINATES IN FAULT SYSTEM
      DISL1-DISL3    : STRIKE-, DIP-, TENSILE-DISLOCATIONS
    OUTPUT
      U(12) : DISPLACEMENT AND THEIR DERIVATIVES
    """
    global ALP1, ALP2, ALP3, ALP4, ALP5, SD, CD, SDSD, CDCD, SDCD, S2D, C2D
    global XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32
    global EY, EZ, FY, FZ, GY, GZ, HY, HZ
    
    F0, F1, F2, F3, PI2 = 0.0, 1.0, 2.0, 3.0, 6.283185307179586
    
    DU = [0.0] * 12

    # Initial computations
    C = D + Z
    X53 = (8.0 * R2 + 9.0 * R * XI + F3 * XI2) * X11**3 / R2
    Y53 = (8.0 * R2 + 9.0 * R * ET + F3 * ET2) * Y11**3 / R2
    H = Q * CD - Z
    Z32 = SD / R3 - H * Y32
    Z53 = F3 * SD / R5 - H * Y53
    Y0 = Y11 - XI2 * Y32
    Z0 = Z32 - XI2 * Z53
    PPY = CD / R3 + Q * Y32 * SD
    PPZ = SD / R3 - Q * Y32 * CD
    QQ = Z * Y32 + Z32 + Z0
    QQY = F3 * C * D / R5 - QQ * SD
    QQZ = F3 * C * Y / R5 - QQ * CD + Q * Y32
    XY = XI * Y11
    QX = Q * X11
    QY = Q * Y11
    QR = F3 * Q / R5
    CQX = C * Q * X53
    CDR = (C + D) / R3
    YY0 = Y / R3 - Y0 * CD
    
    # Initialize U to zeros
    for i in range(12):
        U[i] = F0

    # Strike-slip contribution
    if DISL1 != F0:
        DU[0]  = ALP4 * XY * CD - ALP5 * XI * Q * Z32
        DU[1]  = ALP4 * (CD / R + F2 * QY * SD) - ALP5 * C * Q / R3
        DU[2]  = ALP4 * QY * CD - ALP5 * (C * ET / R3 - Z * Y11 + XI2 * Z32)
        DU[3]  = ALP4 * Y0 * CD - ALP5 * Q * Z0
        DU[4]  = -ALP4 * XI * (CD / R3 + F2 * Q * Y32 * SD) + ALP5 * C * XI * QR
        DU[5]  = -ALP4 * XI * Q * Y32 * CD + ALP5 * XI * (F3 * C * ET / R5 - QQ)
        DU[6]  = -ALP4 * XI * PPY * CD - ALP5 * XI * QQY
        DU[7]  = ALP4 * F2 * (D / R3 - Y0 * SD) * SD - Y / R3 * CD - ALP5 * (CDR * SD - ET / R3 - C * Y * QR)
        DU[8]  = -ALP4 * Q / R3 + YY0 * SD + ALP5 * (CDR * CD + C * D * QR - (Y0 * CD + Q * Z0) * SD)
        DU[9]  = ALP4 * XI * PPZ * CD - ALP5 * XI * QQZ
        DU[10] = ALP4 * F2 * (Y / R3 - Y0 * CD) * SD + D / R3 * CD - ALP5 * (CDR * CD + C * D * QR)
        DU[11] = YY0 * CD - ALP5 * (CDR * SD - C * Y * QR - Y0 * SDSD + Q * Z0 * CD)
        
        for i in range(12):
            U[i] += DISL1 / PI2 * DU[i]

    # Dip-slip contribution
    if DISL2 != F0:
        DU[0]  = ALP4 * CD / R - QY * SD - ALP5 * C * Q / R3
        DU[1]  = ALP4 * Y * X11 - ALP5 * C * ET * Q * X32
        DU[2]  = -D * X11 - XY * SD - ALP5 * C * (X11 - Q2 * X32)
        DU[3]  = -ALP4 * XI / R3 * CD + ALP5 * C * XI * QR + XI * Q * Y32 * SD
        DU[4]  = -ALP4 * Y / R3 + ALP5 * C * ET * QR
        DU[5]  = D / R3 - Y0 * SD + ALP5 * C / R3 * (F1 - F3 * Q2 / R2)
        DU[6]  = -ALP4 * ET / R3 + Y0 * SDSD - ALP5 * (CDR * SD - C * Y * QR)
        DU[7]  = ALP4 * (X11 - Y * Y * X32) - ALP5 * C * ((D + F2 * Q * CD) * X32 - Y * ET * Q * X53)
        DU[8]  = XI * PPY * SD + Y * D * X32 + ALP5 * C * ((Y + F2 * Q * SD) * X32 - Y * Q2 * X53)
        DU[9]  = -Q / R3 + Y0 * SDCD - ALP5 * (CDR * CD + C * D * QR)
        DU[10] = ALP4 * Y * D * X32 - ALP5 * C * ((Y - F2 * Q * SD) * X32 + D * ET * Q * X53)
        DU[11] = -XI * PPZ * SD + X11 - D * D * X32 - ALP5 * C * ((D - F2 * Q * CD) * X32 - D * Q2 * X53)
        
        for i in range(12):
            U[i] += DISL2 / PI2 * DU[i]

    # Tensile-fault contribution
    if DISL3 != F0:
        DU[0]  = -ALP4 * (SD / R + QY * CD) - ALP5 * (Z * Y11 - Q2 * Z32)
        DU[1]  = ALP4 * F2 * XY * SD + D * X11 - ALP5 * C * (X11 - Q2 * X32)
        DU[2]  = ALP4 * (Y * X11 + XY * CD) + ALP5 * Q * (C * ET * X32 + XI * Z32)
        DU[3]  = ALP4 * XI / R3 * SD + XI * Q * Y32 * CD + ALP5 * XI * (F3 * C * ET / R5 - F2 * Z32 - Z0)
        DU[4]  = ALP4 * F2 * Y0 * SD - D / R3 + ALP5 * C / R3 * (F1 - F3 * Q2 / R2)
        DU[5]  = -ALP4 * YY0 - ALP5 * (C * ET * QR - Q * Z0)
        DU[6]  = ALP4 * (Q / R3 + Y0 * SDCD) + ALP5 * (Z / R3 * CD + C * D * QR - Q * Z0 * SD)
        
        for i in range(12):
            U[i] += DISL3 / PI2 * DU[i]

    return


def DCCON0(ALPHA, DIP):
    """
    *******************************************************************
    *****   CALCULATE MEDIUM CONSTANTS AND FAULT-DIP CONSTANTS    *****
    *******************************************************************
    
    INPUT
      ALPHA : MEDIUM CONSTANT  (LAMBDA+MYU)/(LAMBDA+2*MYU)
      DIP   : DIP-ANGLE (DEGREE)
    
    ### CAUTION ### IF COS(DIP) IS SUFFICIENTLY SMALL, IT IS SET TO ZERO
    """
    global ALP1, ALP2, ALP3, ALP4, ALP5, SD, CD, SDSD, CDCD, SDCD, S2D, C2D
    F0, F1, F2, PI2 = 0.0, 1.0, 2.0, 6.283185307179586
    EPS = 1.0e-6

    # Calculating medium constants
    ALP1 = (F1 - ALPHA) / F2
    ALP2 = ALPHA / F2
    ALP3 = (F1 - ALPHA) / ALPHA
    ALP4 = F1 - ALPHA
    ALP5 = ALPHA

    # Convert DIP from degrees to radians and calculate sin/cos
    P18 = PI2 / 360.0
    SD = math.sin(DIP * P18)
    CD = math.cos(DIP * P18)
    
    # Handle small cosine values
    if abs(CD) < EPS:
        CD = F0
        if SD > F0:
            SD = F1
        if SD < F0:
            SD = -F1

    SDSD = SD * SD
    CDCD = CD * CD
    SDCD = SD * CD
    S2D = F2 * SDCD
    C2D = CDCD - SDSD

    return


def DCCON2(XI, ET, Q, SD, CD, KXI, KET):
    """
    **********************************************************************
    *****   CALCULATE STATION GEOMETRY CONSTANTS FOR FINITE SOURCE   *****
    **********************************************************************
    
    INPUT
      XI, ET, Q  : STATION COORDINATES IN FAULT SYSTEM
      SD, CD     : SIN, COS OF DIP-ANGLE
      KXI, KET   : KXI=1, KET=1 MEANS R+XI<EPS, R+ET<EPS, RESPECTIVELY
    
    ### CAUTION ### IF XI, ET, Q ARE SUFFICIENTLY SMALL, THEY ARE SET TO ZERO
    """
    global XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ
    F0, F1, F2, EPS = 0.0, 1.0, 2.0, 1.0e-6

    # Handle small values
    if abs(XI) < EPS:
        XI = F0
    if abs(ET) < EPS:
        ET = F0
    if abs(Q) < EPS:
        Q = F0
    
    XI2 = XI * XI
    ET2 = ET * ET
    Q2 = Q * Q
    R2 = XI2 + ET2 + Q2
    R = R2**0.5
    
    if R == F0:
        return
    
    R3 = R * R2
    R5 = R3 * R2
    Y = ET * CD + Q * SD
    D = ET * SD - Q * CD
    
    if Q == F0:
        TT = F0
    else:
        TT = math.atan(XI * ET / (Q * R))
    
    if KXI == 1:
        ALX = -math.log(R - XI)
        X11 = F0
        X32 = F0
    else:
        RXI = R + XI
        ALX = math.log(RXI)
        X11 = F1 / (R * RXI)
        X32 = (R + RXI) * X11 * X11 / R
    
    if KET == 1:
        ALE = -math.log(R - ET)
        Y11 = F0
        Y32 = F0
    else:
        RET = R + ET
        ALE = math.log(RET)
        Y11 = F1 / (R * RET)
        Y32 = (R + RET) * Y11 * Y11 / R
    
    EY = SD / R - Y * Q / R3
    EZ = CD / R + D * Q / R3
    FY = D / R3 + XI2 * Y32 * SD
    FZ = Y / R3 + XI2 * Y32 * CD
    GY = F2 * X11 * SD - Y * Q * X32
    GZ = F2 * X11 * CD + D * Q * X32
    HY = D * Q * X32 + XI * Q * Y32 * SD
    HZ = Y * Q * X32 + XI * Q * Y32 * CD

    return

