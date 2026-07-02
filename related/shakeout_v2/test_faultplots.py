# Eric Lindsey, 04/2016

import fault_model
import fault_plots
import numpy as np

lat=np.array([-6,6,0,-5.9])
lon=np.array([105,95,104,105.1])
elev=0*lat
latp=np.array([-3,-2,1])
lonp=np.array([100,101,101])
elevp=0*latp

Fplot=fault_plots.FaultPlot3D()
#Fplot.plot_outlines(lon,lat,elev)
Fplot.plot_symbols(lonp,latp,elevp,'ob')
Fplot.plot_shapefile('sumatra')

Fmodel=fault_model.FaultModel()
fname='fault_data_2b_comcot_1_7.txt'
Fmodel.load_patches_comcot(fname)
#fname='SMT_sample.out'
#fname='SMT_10km2_3D_subfaults.out'
#Fmodel.load_patches_comcot(fname)

Fplot.plot_slip_patches(Fmodel,Fmodel.strikeid)

Fplot.set_lims([98,104],[-4,4],[-4,400])
Fplot.showmap()

