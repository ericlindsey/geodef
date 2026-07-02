# Eric Lindsey, 04/2016

import fault_model
import fault_plots
import geod_transform
import numpy as np

fname=('fault_data_2b_comcot_1_7.txt')
#fname=('SMT_sample.out')

F=fault_model.FaultModel()
F.load_patches_comcot(fname)
#lat,lon,depth,strike,dip,leng,wid = F.load_locked_faults(fname)

slip=F.strikeid
x,y,z=geod_transform.geod2enu(F.latc,F.lonc,F.depth,F.latc[0],F.lonc[0],F.depth[0])
fault_plots.plotfaultpatches(x,y,F.depth,slip,F.strike,F.dip,F.L,F.W)

print(F.patchid)
