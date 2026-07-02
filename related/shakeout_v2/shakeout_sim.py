# Eric Lindsey, 04/2016

import fault_model
import fault_plots
import numpy as np


#load sites
sugarsites=np.loadtxt('shakeout_data/SuGAR.sites',usecols=(1,2))
sugarlon=sugarsites[:,0]
sugarlat=sugarsites[:,1]
sugarelev=0*sugarlat

sumosites=np.loadtxt('shakeout_data/SuMo.sites',usecols=(1,2))
sumolon=sumosites[:,0]
sumolat=sumosites[:,1]
sumoelev=0*sumolat

#load insar locations


#load fault model
Fmodel=fault_model.FaultModel()
fname='shakeout_data/mentawai_10km2_3D_subfaults.out'
Fmodel.load_patches_topleft(fname)

#fname='shakeout_data/fault_data_2b_comcot_1_7_edit.txt'
#Fmodel.load_patches_center(fname)

#create greens functions based on obs. locations
sugarGfname='shakeout_data/sugar_greens_mentawai.npy'
sumoGfname='shakeout_data/sumo_greens_mentawai.npy'
recreateG=1
if recreateG==1:
    sugarG=Fmodel.get_greens(sugarlat,sugarlon)
    sumoG=Fmodel.get_greens(sumolat,sumolon)
    np.save(sugarGfname,sugarG)
    np.save(sumoGfname,sumoG)
else:
    sugarG=np.load(sugarGfname)
    sumoG=np.load(sumoGfname)

#load earthquake slip history


#create synthetic displacements


#add synthetic noise



#???teleseis

#aftershocks


#plotting commands.
#create the plot
Fplot=fault_plots.FaultPlot3D()
#plot gps sites
Fplot.plot_symbols(sugarlon,sugarlat,sugarelev,'ob')
Fplot.plot_symbols(sumolon,sumolat,sumoelev,'or')

#plot gps vectors

#plot insar outlines, ?

#plot trench axis
trench=np.loadtxt('shakeout_data/sunda_trench.xy',comments='>')
trenchlon=trench[:,0]
trenchlat=trench[:,1]
trenchelev=trench[:,2]
Fplot.plot_outlines(trenchlon,trenchlat,trenchelev,'--k')

#plot sumatran fault
sf=np.loadtxt('shakeout_data/great_sumatra_fault_nan.xy')
sflon=sf[:,0]
sflat=sf[:,1]
sfelev=0*sflon
Fplot.plot_outlines(sflon,sflat,sfelev,'r')

Fplot.plot_shapefile('shakeout_data/sumatra')
Fplot.plot_slip_patches(Fmodel,Fmodel.strikeid)
Fplot.set_lims([98,104],[-4,4],[-4,400])
Fplot.showmap()
