# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 20:31:54 2016

@author: elindsey
"""
import matplotlib.pyplot as plt
import shapefile # requires shapefile.py, from pyshp package

import numpy as np
import fault_model
import fault_plots
import geod_transform
import slip_model

#load sites
#sugarsites=np.loadtxt('shakeout_data/SuGAR.sites',usecols=(1,2))
x,y = np.meshgrid(np.arange(-.9,1,.2),np.arange(-.9,1,.2))
gpslon=x.flatten()
gpslat=y.flatten()
gpselev=0*gpslat

Fmodel=fault_model.FaultModel()
Smodel=slip_model.SlipModel()
Smodel.set_fault_model(Fmodel)

#patch info
latc=0
lonc=0
depth=100000
strike=-40
dip=20
L=100000
W=50000
Smodel.F.add_patch(latc,lonc,depth,strike,dip,L,W,0,0)
Smodel.F.add_patch(latc,lonc,depth+W,strike,dip,L,W,0,1)


Smodel.slip=np.array([10,10])
Smodel.rake=np.array([90,90])

#compute greens functions and displacements - GPS
gpseast,gpsnorth,gpsup=Smodel.forward_model_static(gpslat,gpslon)  


#%%
plt.figure(1)
ax = plt.gca()
ax.quiver(gpslon,gpslat,gpseast,gpsnorth,angles='xy',scale_units='xy')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
plt.draw()
plt.show()


#ax.set_xlim([98,104])
#ax.set_ylim([-4,4])
#ax.axis('equal')

#fname='shakeout_data/sumatra'
#note, if you need to crop the shapefile use ogr2ogr, e.g.:
#ogr2ogr -f "ESRI Shapefile" <output>.shp <input>.shp -clipsrc 94 107 -6 6
#shp = shapereader.Reader('./data/GSHHS/bts_GSHHS_f_L1')
#for record, geometry in zip(shp.records(), shp.geometries()):
#    self.ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='lightgray',edgecolor='black')
#coast = shapefile.Reader(fname)     
#for shape in coast.shapes():   
#    x, y = zip(*shape.points)
#    ax.plot(x,y,color='k')


