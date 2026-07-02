#!/usr/local/bin/python
# unit tests for okada dislocation functions
# Eric Lindsey, 01/2014

import unittest
import okada85
import numpy as np

class TestOkada85(unittest.TestCase):
    def setUp(self):
        self.nu = 0.25
        self.L  = 3.
        self.W  = 2.
        self.strike = 90.
        self.x      = [ 2.,  2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.]
        self.y      = [ 3.,  3.,  3.,  0.,  0.,  0.,  0.,  0.,  0.]
        self.d      = [ 4.,  4.,  4.,  4.,  4.,  4.,  6.,  6.,  6.]
        self.dip    = [70., 70., 70., 90., 90., 90., 90., 90., 90.]
        self.rake   = [ 0., 90.,  0.,  0., 90.,  0.,180., 90.,180.]
        self.slip   = [ 1.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,  0.]
        self.u3     = [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.]
        self.testresults=[
        (-8.689165004256261e-03, -4.297582189741731e-03, -2.747405827638823e-03, -1.220438675268007e-03, 2.469697394431684e-04, -8.191372879334214e-03, -5.813975227325929e-04, -5.174968695670765e-03, 2.945389615109786e-04),
        (-4.682348762835457e-03,         -3.526726796871769e-02,         -3.563855767326866e-02,         -8.867245527911540e-03,         -1.518582321831328e-04,         4.056585617604535e-03 ,         -1.035487654241981e-02,         4.088128489997486e-03 ,         2.626254787455854e-03),
        (-2.659960096441058e-04,         1.056407487698295e-02 ,         3.214193114221133e-03 ,         -5.654954762127719e-04,         1.992743608041701e-03 ,         -1.066213796012181e-03,         1.229710985665510e-02 ,         -3.730219351653117e-04,         1.040095554700254e-02),
        (0.000000000000000e+00 ,         5.253097376910021e-03 ,         0.000000000000000e+00 ,         0.000000000000000e+00 ,         -1.863722792616869e-02,         -2.325128307637018e-03,         0.000000000000000e+00 ,         0.000000000000000e+00 ,         2.288515018789224e-02),
        (0.000000000000000e+00 ,         0.000000000000000e+00 ,         0.000000000000000e+00 ,         0.000000000000000e+00 ,         2.747808530970814e-02 ,         0.000000000000000e+00 ,         0.000000000000000e+00 ,         0.000000000000000e+00 ,         -7.166487554729019e-02),
        (1.222848229982107e-02 ,         0.000000000000000e+00 ,         -1.606274646428264e-02,         -4.181651077400775e-03,         0.000000000000000e+00 ,         0.000000000000000e+00 ,         -2.325128307637018e-03,         -9.146107533038163e-03,         0.000000000000000e+00),
        (0.000000000000000e+00 ,         -1.303097451419518e-03,         0.000000000000000e+00 ,         0.000000000000000e+00 ,         2.726036397973775e-03 ,         7.345401310885204e-04 ,         0.000000000000000e+00 ,         0.000000000000000e+00 ,         -4.421657134187061e-03),
        (0.000000000000000e+00 ,         0.000000000000000e+00 ,         0.000000000000000e+00 ,         0.000000000000000e+00 ,         5.157341419851447e-03 ,         0.000000000000000e+00 ,         0.000000000000000e+00 ,         0.000000000000000e+00 ,         -1.901154674368966e-02),
        (3.506717844685072e-03 ,         0.000000000000000e+00 ,         -7.740131086250291e-03,         -1.770218903898004e-03,         0.000000000000000e+00 ,         0.000000000000000e+00 ,         -7.345401310885204e-04,         -1.842986424261335e-03,         0.000000000000000e+00)
        ]
        
    def test_cases(self):
        self.setUp()
        for i in range(len(self.x)):
            # arguments (e, n, depth, strike, dip, L, W, rake, slip, open, nu)
            ue,un,uz = okada85.displacement(
                self.x[i]-self.L/2.,
                self.y[i]-np.cos(self.dip[i]*np.pi/180.)*self.W/2.,
                self.d[i]-np.sin(self.dip[i]*np.pi/180.)*self.W/2.,
                self.strike, self.dip[i],
                self.L, self.W,
                self.rake[i], self.slip[i], self.u3[i],self.nu)
            uze,uzn = okada85.tilt(
                self.x[i]-self.L/2.,
                self.y[i]-np.cos(self.dip[i]*np.pi/180.)*self.W/2.,
                self.d[i]-np.sin(self.dip[i]*np.pi/180.)*self.W/2.,
                self.strike, self.dip[i],
                self.L, self.W,
                self.rake[i], self.slip[i], self.u3[i],self.nu)
            unn,une,uen,uee = okada85.strain(
                self.x[i]-self.L/2.,
                self.y[i]-np.cos(self.dip[i]*np.pi/180.)*self.W/2.,
                self.d[i]-np.sin(self.dip[i]*np.pi/180.)*self.W/2.,
                self.strike, self.dip[i],
                self.L, self.W,
                self.rake[i], self.slip[i], self.u3[i],self.nu)
            resulttuple=(ue,un,uz,-uee,-uen,-une,-unn,uze,uzn)
            testtuple=self.testresults[i]
            for j in range(len(resulttuple)):
                self.assertAlmostEqual(testtuple[j],resulttuple[j],places=15)

suite = unittest.TestLoader().loadTestsFromTestCase(TestOkada85)
unittest.TextTestRunner(verbosity=2).run(suite)

