#!/usr/bin/env python

from okada_wrapper import dc3dwrapper, dc3d0wrapper


def do_main():
    """
    Assumes the fault is striking at 0 degrees.
    (usually I have a rotation matrix to get an arbitrary fault into north-striking,
    but I'm trying to simplify the function call for the test).
    """
    strike_slip = 1.2    # dc3d coordinate system has left-lateral positive.
    reverse_slip = -0.3
    tensile_slip = 0
    W = 4  # km
    L = 10  # km
    top = 2  # km
    dip_angle = 78  # degrees
    alpha = 2/3  # consistent with poisson's ratio of 0.25 I think
    calc_xyz = [19, 24, -6]   # target point, in km from the corner of the fault, negative means down
    success, u, grad_u = dc3dwrapper(alpha, [calc_xyz[0], calc_xyz[1], calc_xyz[2]], top, dip_angle,
                                     [0, L], [-W, 0],
                                     [strike_slip, reverse_slip, tensile_slip])
    grad_u = grad_u * 1e-3  # DC3D Unit correction.
    print("Success:", success)
    print("Displacements (m):", u)
    print("Displacement gradients:", grad_u)
    return


"""
Response on KZM Computer, September 25, 2024:

Success: 0
Displacements (m): [-0.0053241  -0.00605796  0.00257908]
Displacement gradients: [[-8.24639792e-08 -7.20125026e-08  3.84831619e-08]
 [ 4.15611605e-07  4.48167644e-07 -1.76400456e-07]
 [-1.20750374e-07 -9.79826145e-08 -1.10257453e-07]]
"""
