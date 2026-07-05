# okada85.py
# python routines for calculating surface displacements and strains
# due to a finite rectangular dislocation source.

            #conversion ToDo
            # R appears in every single function... reuse?
#
# Requirements: numpy (numerical python package)
#
# Included routines:
# ue,un,uz =  displacement(e,n,depth,strike,dip,L,W,rake,slip,open,nu)
# uze,uzn =           tilt(e,n,depth,strike,dip,L,W,rake,slip,open,nu)
# unn,une,uen,uee = strain(e,n,depth,strike,dip,L,W,rake,slip,open,nu)
#
# e,n             : surface coordinates relative to the fault centroid
# depth           : depth of the fault centroid (depth > 0)
# strike          : in degrees from North, 90 points East.
# dip             : in degrees from horizontal, to the right side of the trace
# rake            : in degrees, direction the hanging wall moves. If slip > 0,
#                 : a rake of 0 is left-lateral slip, +90 is reverse fault.
# L,W             : length (along-strike) and width (along-dip) of the fault patch
# slip, open      : displacements, uniform across entire patch
# nu              : poisson's ratio (optional, default 0.25)
#
# Units of e,n,depth,L,W,slip,open should be the same (eg. everything in meters)
#
# For strains, note
#    POSITIVE = COMPRESSION
# Note that vertical strain components can be obtained with following equations:
#    uNZ = -uZN;
#    uEZ = -uZE;
#    uZZ = -(uEE + uNN)*NU/(1-NU);
#
# References:
#   Aki K., and P. G. Richards, Quantitative seismology, Freemann & Co,
#       New York, 1980.
#   Okada Y., Surface deformation due to shear and tensile faults in a
#       half-space, Bull. Seismol. Soc. Am., 75:4, 1135-1154, 1985.
#
# =================================================================
# Licensing / Copyright
#
# Copyright (c) 2014, Eric Lindsey, covered by BSD License (see text below).
# All rights reserved.
#
# Version history:
# [01/2014] Converted to python from [08/2012] matlab version by Francois Beauducel:
# available http://www.ipgp.fr/~beaudu/matlab.html
#
# Matlab version:
# Copyright (c) 1997-2012, Francois Beauducel, covered by BSD License (see text below).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# =================================================================

from geodef.backend import xp


# =================================================================
def setup_args(e,n,depth,strike,dip,L,W,rake,slip,open):

    # ensure array-like behavior
    e=xp.array(e)
    n=xp.array(n)
    depth=xp.array(depth)
    L=xp.array(L)
    W=xp.array(W)
    slip=xp.array(slip)
    open=xp.array(open)

    #convert to radians and ensure array-like behavior
    strike = xp.array(strike*xp.pi/180)
    dip = xp.array(dip*xp.pi/180)
    rake = xp.array(rake*xp.pi/180)

    # Defines dislocation in the fault plane system
    # 's' denotes slip scaled by the pre-multiplier 2*pi
    U1s = xp.cos(rake)*slip/(2*xp.pi)
    U2s = xp.sin(rake)*slip/(2*xp.pi)
    U3s = open/(2*xp.pi)

    # Converts fault coordinates (E,N,DEPTH) relative to centroid
    # into Okada's reference system (X,Y,D)
    d = depth + xp.sin(dip)*W/2    # d is fault's top edge
    ec = e + xp.cos(strike)*xp.cos(dip)*W/2
    nc = n - xp.sin(strike)*xp.cos(dip)*W/2
    x = xp.cos(strike)*nc + xp.sin(strike)*ec + L/2
    y = xp.sin(strike)*nc - xp.cos(strike)*ec + xp.cos(dip)*W

    # Variable substitution (independent from xi and eta)
    p = y*xp.cos(dip) + d*xp.sin(dip)
    q = y*xp.sin(dip) - d*xp.cos(dip)

    return x,p,L,W,q,strike,dip,U1s,U2s,U3s


# =================================================================
def arctan_term(xi,eta,q,R):
    # Okada's arctan(xi*eta/(q*R)) term [equations (25)-(27) p. 1144],
    # rewritten so automatic differentiation stays finite when q == 0
    # (observation on the fault-plane ray, where the naive form divides
    # by zero and its derivative evaluates 0*inf = nan). Whichever of
    # num/den is larger goes in the denominator, via the identity
    # arctan(u) = sign(u)*pi/2 - arctan(1/u); at q == 0 exactly this
    # yields Okada's prescribed value of 0 for the term.
    num = xi*eta
    den = q*R
    big = xp.abs(num) > xp.abs(den)
    den_direct = xp.where(big | (den == 0), 1.0, den)
    num_inverse = xp.where(big, num, 1.0)
    direct = xp.arctan(num/den_direct)
    inverse = xp.sign(num)*xp.sign(den)*xp.pi/2 - xp.arctan(den/num_inverse)
    return xp.where(big, inverse, direct)


# =================================================================
def displacement(e,n,depth,strike,dip,L,W,rake,slip,open,nu=0.25):

    #convert coordinate systems, radians, ensure array_like, etc.
    x,p,L,W,q,strike,dip,U1s,U2s,U3s = setup_args(e,n,depth,strike,dip,L,W,rake,slip,open)

    ux = ( -U1s * chinnery(ux_ss,x,p,L,W,q,dip,nu)   # strike-slip
           -U2s * chinnery(ux_ds,x,p,L,W,q,dip,nu)   # dip-slip
           +U3s * chinnery(ux_tf,x,p,L,W,q,dip,nu) ) # tensile fault

    uy = ( -U1s * chinnery(uy_ss,x,p,L,W,q,dip,nu)   # strike-slip
           -U2s * chinnery(uy_ds,x,p,L,W,q,dip,nu)   # dip-slip
           +U3s * chinnery(uy_tf,x,p,L,W,q,dip,nu) ) # tensile fault

    uz = ( -U1s * chinnery(uz_ss,x,p,L,W,q,dip,nu)   # strike-slip
           -U2s * chinnery(uz_ds,x,p,L,W,q,dip,nu)   # dip-slip
           +U3s * chinnery(uz_tf,x,p,L,W,q,dip,nu) ) # tensile fault

    # Rotation from Okada's axes to geographic
    ue = xp.sin(strike)*ux - xp.cos(strike)*uy
    un = xp.cos(strike)*ux + xp.sin(strike)*uy

    # function call to pick these up looks like dE,dN,dU = displacement(...)
    return ue,un,uz


# =================================================================
def tilt(e,n,depth,strike,dip,L,W,rake,slip,open,nu=0.25):

    #convert coordinate systems, radians, ensure array_like, etc.
    x,p,L,W,q,strike,dip,U1s,U2s,U3s = setup_args(e,n,depth,strike,dip,L,W,rake,slip,open)

    uzx = ( -U1s * chinnery(uzx_ss,x,p,L,W,q,dip,nu)   # strike-slip
            -U2s * chinnery(uzx_ds,x,p,L,W,q,dip,nu)   # dip-slip
            +U3s * chinnery(uzx_tf,x,p,L,W,q,dip,nu) ) # tensile fault

    uzy = ( -U1s * chinnery(uzy_ss,x,p,L,W,q,dip,nu)   # strike-slip
            -U2s * chinnery(uzy_ds,x,p,L,W,q,dip,nu)   # dip-slip
            +U3s * chinnery(uzy_tf,x,p,L,W,q,dip,nu) ) # tensile fault

    # Rotation from Okada's axes to geographic
    uze = -xp.sin(strike)*uzx + xp.cos(strike)*uzy
    uzn = -xp.cos(strike)*uzx - xp.sin(strike)*uzy

    return uze,uzn


# =================================================================
def strain(e,n,depth,strike,dip,L,W,rake,slip,open,nu=0.25):
    # positive = compression

    #convert coordinate systems, radians, ensure array_like, etc.
    x,p,L,W,q,strike,dip,U1s,U2s,U3s = setup_args(e,n,depth,strike,dip,L,W,rake,slip,open)

    uxx = ( -U1s * chinnery(uxx_ss,x,p,L,W,q,dip,nu)   # strike-slip
            -U2s * chinnery(uxx_ds,x,p,L,W,q,dip,nu)   # dip-slip
            +U3s * chinnery(uxx_tf,x,p,L,W,q,dip,nu) ) # tensile fault

    uxy = ( -U1s * chinnery(uxy_ss,x,p,L,W,q,dip,nu)   # strike-slip
            -U2s * chinnery(uxy_ds,x,p,L,W,q,dip,nu)   # dip-slip
            +U3s * chinnery(uxy_tf,x,p,L,W,q,dip,nu) ) # tensile fault

    uyx = ( -U1s * chinnery(uyx_ss,x,p,L,W,q,dip,nu)   # strike-slip
            -U2s * chinnery(uyx_ds,x,p,L,W,q,dip,nu)   # dip-slip
            +U3s * chinnery(uyx_tf,x,p,L,W,q,dip,nu) ) # tensile fault

    uyy = ( -U1s * chinnery(uyy_ss,x,p,L,W,q,dip,nu)   # strike-slip
            -U2s * chinnery(uyy_ds,x,p,L,W,q,dip,nu)   # dip-slip
            +U3s * chinnery(uyy_tf,x,p,L,W,q,dip,nu) ) # tensile fault

    # Rotation from Okada's axes to geographic
    unn =  xp.square(xp.cos(strike))*uxx + xp.sin(2*strike)*(uxy + uyx)/2 + xp.square(xp.sin(strike))*uyy
    une =  xp.square(xp.sin(strike))*uyx + xp.sin(2*strike)*(uxx - uyy)/2 - xp.square(xp.cos(strike))*uxy
    uen = -xp.square(xp.cos(strike))*uyx + xp.sin(2*strike)*(uxx - uyy)/2 + xp.square(xp.sin(strike))*uxy
    uee =  xp.square(xp.sin(strike))*uxx - xp.sin(2*strike)*(uyx + uxy)/2 + xp.square(xp.cos(strike))*uyy

    #note, une, uen should be the same?
    return unn,une,uen,uee


# =================================================================
# Chinnery's notation [equation (24) p. 1143]
#
# -----------------------------------------------------------------
def chinnery(f,x,p,L,W,q,dip,nu):
    u = f(x,p,q,dip,nu) - f(x,p-W,q,dip,nu) - f(x-L,p,q,dip,nu) + f(x-L,p-W,q,dip,nu)
    return u


# =================================================================
# Displacement subfunctions

# strike-slip displacement subfunctions [equation (25) p. 1144]
# -----------------------------------------------------------------
def ux_ss(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    u = xi*q/(R*(R + eta)) + arctan_term(xi,eta,q,R) + I1(xi,eta,q,dip,nu,R)*xp.sin(dip)
    return u

# -----------------------------------------------------------------
def uy_ss(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    u = (eta*xp.cos(dip) + q*xp.sin(dip))*q/(R*(R + eta)) + q*xp.cos(dip)/(R + eta) + I2(eta,q,dip,nu,R)*xp.sin(dip)
    return u

# -----------------------------------------------------------------
def uz_ss(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    u = (eta*xp.sin(dip) - q*xp.cos(dip))*q/(R*(R + eta)) + q*xp.sin(dip)/(R + eta) + I4(db,eta,q,dip,nu,R)*xp.sin(dip)
    return u

# dip-slip displacement subfunctions [equation (26) p. 1144]
# -----------------------------------------------------------------
def ux_ds(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    u = q/R - I3(eta,q,dip,nu,R)*xp.sin(dip)*xp.cos(dip)
    return u

# -----------------------------------------------------------------
def uy_ds(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    u = (eta*xp.cos(dip) + q*xp.sin(dip))*q/(R*(R + xi)) + xp.cos(dip)*arctan_term(xi,eta,q,R) - I1(xi,eta,q,dip,nu,R)*xp.sin(dip)*xp.cos(dip)
    return u

# -----------------------------------------------------------------
def uz_ds(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    u = db*q/(R*(R + xi)) + xp.sin(dip)*arctan_term(xi,eta,q,R) - I5(xi,eta,q,dip,nu,R,db)*xp.sin(dip)*xp.cos(dip)
    return u

# tensile fault displacement subfunctions [equation (27) p. 1144]
# -----------------------------------------------------------------
def ux_tf(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    u = xp.square(q) /(R*(R + eta)) - I3(eta,q,dip,nu,R)*xp.square(xp.sin(dip))
    return u

# -----------------------------------------------------------------
def uy_tf(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    u = -(eta*xp.sin(dip) - q*xp.cos(dip))*q/(R*(R + xi)) - xp.sin(dip)*(xi*q/(R*(R + eta)) - arctan_term(xi,eta,q,R)) - I1(xi,eta,q,dip,nu,R)*xp.square(xp.sin(dip))
    return u

# -----------------------------------------------------------------
def uz_tf(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    u = (eta*xp.cos(dip) + q*xp.sin(dip))*q/(R*(R + xi)) + xp.cos(dip)*(xi*q/(R*(R + eta)) - arctan_term(xi,eta,q,R)) - I5(xi,eta,q,dip,nu,R,db)*xp.square(xp.sin(dip))
    return u


# I... displacement subfunctions [equations (28) (29) p. 1144-1145]
# -----------------------------------------------------------------
def I1(xi,eta,q,dip,nu,R):
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    I = xp.where(xp.cos(dip) > xp.spacing(1),
        (1 - 2*nu) * (-xi/(xp.cos(dip)*(R + db))) - xp.sin(dip)/xp.cos(dip) *I5(xi,eta,q,dip,nu,R,db),
        -(1 - 2*nu)/2 * xi*q/xp.square(R + db) )
    return I

# -----------------------------------------------------------------
def I2(eta,q,dip,nu,R):
    I = (1 - 2*nu) * (-xp.log(R + eta)) - I3(eta,q,dip,nu,R)
    return I

# -----------------------------------------------------------------
def I3(eta,q,dip,nu,R):
    yb = eta*xp.cos(dip) + q*xp.sin(dip)
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    I = xp.where(xp.cos(dip) > xp.spacing(1),
        (1 - 2*nu) * (yb/(xp.cos(dip)*(R + db)) - xp.log(R + eta)) + xp.sin(dip)/xp.cos(dip) * I4(db,eta,q,dip,nu,R),
        (1 - 2*nu)/2 * (eta/(R + db) + yb*q/xp.square(R + db) - xp.log(R + eta)) )
    return I

# -----------------------------------------------------------------
def I4(db,eta,q,dip,nu,R):
    I = xp.where(xp.cos(dip) > xp.spacing(1),
        (1 - 2*nu) * 1/xp.cos(dip) * (xp.log(R + db) - xp.sin(dip)*xp.log(R + eta)),
        -(1 - 2*nu) * q/(R + db) )
    return I

# -----------------------------------------------------------------
def I5(xi,eta,q,dip,nu,R,db):
    X = xp.sqrt(xp.square(xi) + xp.square(q))
    # fix a strange intermittent zero-division warning in the xp.where() clause
    xi = xp.where(xi==0, xi+xp.spacing(1), xi)
    I = xp.where(xp.cos(dip) > xp.spacing(1),
        (1 - 2*nu) * 2/xp.cos(dip) * xp.arctan((eta*(X + q*xp.cos(dip)) + X*(R + X)*xp.sin(dip))/(xi*(R + X)*xp.cos(dip))),
        -(1 - 2*nu) * xi*xp.sin(dip)/(R + db) )
    return I


# =================================================================
# Tilt subfunctions

# strike-slip tilt subfunctions [equation (37) p. 1147]
# -----------------------------------------------------------------
def uzx_ss(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    u = -xi*xp.square(q)*A(eta,R)*xp.cos(dip) + ((xi*q)/xp.power(R,3) - K1(xi,eta,q,dip,nu,R))*xp.sin(dip)
    return u

# -----------------------------------------------------------------
def uzy_ss(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    yb = eta*xp.cos(dip) + q*xp.sin(dip)
    u = (db*q/xp.power(R,3))*xp.cos(dip) + (xp.square(xi)*q*A(eta,R)*xp.cos(dip) - xp.sin(dip)/R + yb*q/xp.power(R,3) - K2(xi,eta,q,dip,nu,R))*xp.sin(dip)
    return u

# dip-slip tilt subfunctions [equation (38) p. 1147]
# -----------------------------------------------------------------
def uzx_ds(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    u = db*q/xp.power(R,3) + q*xp.sin(dip)/(R*(R + eta)) + K3(xi,eta,q,dip,nu,R)*xp.sin(dip)*xp.cos(dip)
    return u

# -----------------------------------------------------------------
def uzy_ds(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    yb = eta*xp.cos(dip) + q*xp.sin(dip)
    u = yb*db*q*A(xi,R) - (2*db/(R*(R + xi)) + xi*xp.sin(dip)/(R*(R + eta)))*xp.sin(dip) + K1(xi,eta,q,dip,nu,R)*xp.sin(dip)*xp.cos(dip)
    return u

# tensile fault tilt subfunctions [equation (39) p. 1147]
# -----------------------------------------------------------------
def uzx_tf(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    u = xp.square(q)/xp.power(R,3)*xp.sin(dip) - xp.power(q,3)*A(eta,R)*xp.cos(dip) + K3(xi,eta,q,dip,nu,R)*xp.square(xp.sin(dip))
    return u

# -----------------------------------------------------------------
def uzy_tf(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    yb = eta*xp.cos(dip) + q*xp.sin(dip)
    u = (yb*xp.sin(dip) + db*xp.cos(dip))*xp.square(q)*A(xi,R) + xi*xp.square(q)*A(eta,R)*xp.sin(dip)*xp.cos(dip) - (2*q/(R*(R + xi)) - K1(xi,eta,q,dip,nu,R))*xp.square(xp.sin(dip))
    return u

# -----------------------------------------------------------------
def A(x,R):
    a = (2*R + x)/(xp.power(R,3)*xp.square(R + x))
    return a

# K... tilt subfunctions [equations (40) (41) p. 1148]
# -----------------------------------------------------------------
def K1(xi,eta,q,dip,nu,R):
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    K = xp.where(xp.cos(dip) > xp.spacing(1),
        (1 - 2*nu) * xi/xp.cos(dip) * (1/(R*(R + db)) - xp.sin(dip)/(R*(R + eta))),
        (1 - 2*nu) * xi*q/xp.square(R + db) )
    return K

# -----------------------------------------------------------------
def K2(xi,eta,q,dip,nu,R):
    K = (1 - 2*nu) * (-xp.sin(dip)/R + q*xp.cos(dip)/(R*(R + eta))) - K3(xi,eta,q,dip,nu,R)
    return K

# -----------------------------------------------------------------
def K3(xi,eta,q,dip,nu,R):
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    yb = eta*xp.cos(dip) + q*xp.sin(dip)
    K = xp.where(xp.cos(dip) > xp.spacing(1),
        (1 - 2*nu) * 1/xp.cos(dip) * (q/(R*(R + eta)) - yb/(R*(R + db))),
        (1 - 2*nu) * xp.sin(dip)/(R + db) * (xp.square(xi)/(R*(R + db)) - 1) )
    return K


# =================================================================
# Strain subfunctions

# strike-slip strain subfunctions [equation (31) p. 1145]
# -----------------------------------------------------------------
def uxx_ss(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    u = xp.square(xi)*q*A(eta,R) - J1(xi,eta,q,dip,nu,R)*xp.sin(dip)
    return u

# -----------------------------------------------------------------
def uxy_ss(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    u = xp.power(xi,3)*db/(xp.power(R,3)*(xp.square(eta) + xp.square(q))) - (xp.power(xi,3)*A(eta,R) + J2(xi,eta,q,dip,nu,R))*xp.sin(dip)
    return u

# -----------------------------------------------------------------
def uyx_ss(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    u = xi*q/xp.power(R,3)*xp.cos(dip) + (xi*xp.square(q)*A(eta,R) - J2(xi,eta,q,dip,nu,R))*xp.sin(dip)
    return u

# -----------------------------------------------------------------
def uyy_ss(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    yb = eta*xp.cos(dip) + q*xp.sin(dip)
    u = yb*q/xp.power(R,3)*xp.cos(dip) + (xp.power(q,3)*A(eta,R)*xp.sin(dip) - 2*q*xp.sin(dip)/(R*(R + eta)) - (xp.square(xi) + xp.square(eta))/xp.power(R,3)*xp.cos(dip) - J4(xi,eta,q,dip,nu,R))*xp.sin(dip)
    return u

# dip-slip strain subfunctions [equation (32) p. 1146]
# -----------------------------------------------------------------
def uxx_ds(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    u = xi*q/xp.power(R,3) + J3(xi,eta,q,dip,nu,R)*xp.sin(dip)*xp.cos(dip)
    return u

# -----------------------------------------------------------------
def uxy_ds(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    yb = eta*xp.cos(dip) + q*xp.sin(dip)
    u = yb*q/xp.power(R,3) - xp.sin(dip)/R + J1(xi,eta,q,dip,nu,R)*xp.sin(dip)*xp.cos(dip)
    return u

# -----------------------------------------------------------------
def uyx_ds(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    yb = eta*xp.cos(dip) + q*xp.sin(dip)
    u = yb*q/xp.power(R,3) + q*xp.cos(dip)/(R*(R + eta)) + J1(xi,eta,q,dip,nu,R)*xp.sin(dip)*xp.cos(dip)
    return u

# -----------------------------------------------------------------
def uyy_ds(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    yb = eta*xp.cos(dip) + q*xp.sin(dip)
    u = xp.square(yb)*q*A(xi,R) - (2*yb/(R*(R + xi)) + xi*xp.cos(dip)/(R*(R + eta)))*xp.sin(dip) + J2(xi,eta,q,dip,nu,R)*xp.sin(dip)*xp.cos(dip)
    return u

# tensile fault strain subfunctions [equation (33) p. 1146]
# -----------------------------------------------------------------
def uxx_tf(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    u = xi*xp.square(q)*A(eta,R) + J3(xi,eta,q,dip,nu,R)*xp.square(xp.sin(dip))
    return u

# -----------------------------------------------------------------
def uxy_tf(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    u = -db*q/xp.power(R,3) - xp.square(xi)*q*A(eta,R)*xp.sin(dip) + J1(xi,eta,q,dip,nu,R)*xp.square(xp.sin(dip))
    return u

# -----------------------------------------------------------------
def uyx_tf(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    u = xp.square(q)/xp.power(R,3)*xp.cos(dip) + xp.power(q,3)*A(eta,R)*xp.sin(dip) + J1(xi,eta,q,dip,nu,R)*xp.square(xp.sin(dip))
    return u

# -----------------------------------------------------------------
def uyy_tf(xi,eta,q,dip,nu):
    R = xp.sqrt(xp.square(xi) + xp.square(eta) + xp.square(q))
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    yb = eta*xp.cos(dip) + q*xp.sin(dip)
    u = (yb*xp.cos(dip) - db*xp.sin(dip))*xp.square(q)*A(xi,R) - q*xp.sin(2*dip)/(R*(R + xi)) - (xi*xp.square(q)*A(eta,R) - J2(xi,eta,q,dip,nu,R))*xp.square(xp.sin(dip))
    return u


# J... tensile fault subfunctions [equations (34) (35) p. 1146-1147]
# -----------------------------------------------------------------
def J1(xi,eta,q,dip,nu,R):
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    J = xp.where(xp.cos(dip) > xp.spacing(1),
        (1 - 2*nu) * 1/xp.cos(dip) * (xp.square(xi)/(R*xp.square(R + db)) - 1/(R + db)) - xp.sin(dip)/xp.cos(dip)*K3(xi,eta,q,dip,nu,R),
        (1 - 2*nu)/2 * q/xp.square(R + db) * (2*xp.square(xi)/(R*(R + db)) - 1) )
    return J

# -----------------------------------------------------------------
def J2(xi,eta,q,dip,nu,R):
    db = eta*xp.sin(dip) - q*xp.cos(dip)
    yb = eta*xp.cos(dip) + q*xp.sin(dip)
    J = xp.where(xp.cos(dip) > xp.spacing(1),
        (1 - 2*nu) * 1/xp.cos(dip) * xi*yb/(R*xp.square(R + db)) - xp.sin(dip)/xp.cos(dip)*K1(xi,eta,q,dip,nu,R),
        (1 - 2*nu)/2 * xi*xp.sin(dip)/xp.square(R + db) * (2*xp.square(q)/(R*(R + db)) - 1) )
    return J

# -----------------------------------------------------------------
def J3(xi,eta,q,dip,nu,R):
    J = (1 - 2*nu) * -xi/(R*(R + eta)) - J2(xi,eta,q,dip,nu,R)
    return J

# -----------------------------------------------------------------
def J4(xi,eta,q,dip,nu,R):
    J = (1 - 2*nu) * (-xp.cos(dip)/R - q*xp.sin(dip)/(R*(R + eta))) - J1(xi,eta,q,dip,nu,R)
    return J

