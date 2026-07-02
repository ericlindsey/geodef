# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:22:56 2016

@author: elindsey
"""

import numpy as np
import fault_model
import fault_plots
import geod_transform
import slip_model
import os
import matplotlib.pyplot as plt

# load the original input slip model
Fmodel=fault_model.FaultModel()
Smodel=slip_model.SlipModel()
Smodel.set_fault_model(Fmodel)
fname='shakeout_data/mentawai_10km2_3d_slipmodel.txt'
Smodel.load_slip_static(fname)

# load a duplicate copy to use for comparison
Fmodel2=fault_model.FaultModel()
Smodel2=slip_model.SlipModel()
Smodel2.set_fault_model(Fmodel2)
Smodel2.load_slip_static(fname)
#clear the slip and rake
Smodel2.rake=np.array([])
Smodel2.slip=np.array([])


# load a duplicate copy to use for comparison
Fmodel3=fault_model.FaultModel()
Smodel3=slip_model.SlipModel()
Smodel3.set_fault_model(Fmodel3)
Smodel3.load_slip_static(fname)
#clear the slip and rake
Smodel3.rake=np.array([])
Smodel3.slip=np.array([])

# load a duplicate copy to use for comparison
Fmodel4=fault_model.FaultModel()
Smodel4=slip_model.SlipModel()
Smodel4.set_fault_model(Fmodel4)
Smodel4.load_slip_static(fname)
#clear the slip and rake
Smodel4.rake=np.array([])
Smodel4.slip=np.array([])

# load a duplicate copy to use for comparison
Fmodel5=fault_model.FaultModel()
Smodel5=slip_model.SlipModel()
Smodel5.set_fault_model(Fmodel5)
Smodel5.load_slip_static(fname)
#clear the slip and rake
Smodel5.rake=np.array([])
Smodel5.slip=np.array([])


# load a duplicate copy to use for comparison
Fmodel6=fault_model.FaultModel()
Smodel6=slip_model.SlipModel()
Smodel6.set_fault_model(Fmodel6)
Smodel6.load_slip_static(fname)
#clear the slip and rake
Smodel6.rake=np.array([])
Smodel6.slip=np.array([])

# project slip from the comparison model onto the real one
#create synthetic slip
#comcot=np.loadtxt('shakeout_data/fault_data_2b_comcot_1_7_edit.txt')
 #   lon  lat   dep(km)  strike   dip   len_km   wid_km   dip-slip(cm)   strike-slip(cm)

#%% load QQ slip

# cLon        cLat     dep km  Len km   Width      Rake      strike     dip     slip (cm)
sinput=np.loadtxt('shakeout_models/inverse_okada_slip_dist_all_MentawaiSO_exp111.txt')
slon=sinput[:,0]
slat=sinput[:,1]
sdep=sinput[:,2]
sslip=sinput[:,8]

#sslip=np.sqrt(seastslip**2 + snorthslip**2)
#srake=np.mod(sstr - np.arctan2(seastslip,snorthslip)*180/np.pi ,360)


#for each patch in the fault model, find the nearest input slip patch and assign its rake and slip.
#Use zero if the patch is more than 20km away from any comcot patch

for i in range(Fmodel.npatches):
    distarray=np.array([])
    for j in range(len(slat)):
        distarray=np.append(distarray,np.sqrt((Fmodel.depth[i] - sdep[j])**2 + geod_transform.haversine(slat[j],slon[j],Fmodel.latc[i],Fmodel.lonc[i])**2))
    #if((Fmodel.depth[i]>20000 and min(distarray)>20000) or min(distarray)>50000):
    #    Smodel.slip=np.append(Smodel.slip,0)
    #else:
    nearestpt=np.where(distarray==min(distarray))[0][0]
    #Smodel2.rake=np.append(Smodel2.rake,srake[nearestpt])
    Smodel2.slip=np.append(Smodel2.slip,sslip[nearestpt])



#%% load Lujia slip
fname='shakeout_models/EQ20160427co_40km2_kp300000_subfaults.out'

Flj=fault_model.FaultModel()
Flj.load_patches_topleft(fname)
slon=Flj.lonc
slat=Flj.latc
sdep=Flj.depth
sinput=np.loadtxt(fname)
srake=sinput[:,10]
sslip=sinput[:,11]



#for each patch in the fault model, find the nearest input slip patch and assign its rake and slip.
#Use zero if the patch is more than 20km away from any comcot patch

for i in range(Fmodel.npatches):
    distarray=np.array([])
    for j in range(len(slat)):
        distarray=np.append(distarray,np.sqrt((Fmodel.depth[i] - sdep[j])**2 + geod_transform.haversine(slat[j],slon[j],Fmodel.latc[i],Fmodel.lonc[i])**2))
    #if((Fmodel.depth[i]>20000 and min(distarray)>20000) or min(distarray)>50000):
    #    Smodel.slip=np.append(Smodel.slip,0)
    #else:
    nearestpt=np.where(distarray==min(distarray))[0][0]
    #Smodel3.rake=np.append(Smodel3.rake,srake[nearestpt])
    Smodel3.slip=np.append(Smodel3.slip,sslip[nearestpt])


#%% load Louisa slip

sinput=np.loadtxt('shakeout_models/pcaim_slip_dist_group_format_PCAIM_lujiaMesh_PC2_LW100_sugar+sumo.txt')
## (1)no (2)dnum (3)snum (4)lon (5)lat (6)z (7)length (8)width (9)strike (10)dip (11)rake (12)slip

slon=sinput[:,3]
slat=sinput[:,4]
sdep=sinput[:,5]
sstr=sinput[:,8]
sdip=sinput[:,9]
srake=sinput[:,10]
sslip=sinput[:,11]


#for each patch in the fault model, find the nearest input slip patch and assign its rake and slip.
#Use zero if the patch is more than 20km away from any comcot patch

for i in range(Fmodel.npatches):
    distarray=np.array([])
    for j in range(len(slat)):
        distarray=np.append(distarray,np.sqrt((Fmodel.depth[i] - sdep[j])**2 + geod_transform.haversine(slat[j],slon[j],Fmodel.latc[i],Fmodel.lonc[i])**2))
    #if((Fmodel.depth[i]>20000 and min(distarray)>20000) or min(distarray)>50000):
    #    Smodel.slip=np.append(Smodel.slip,0)
    #else:
    nearestpt=np.where(distarray==min(distarray))[0][0]
    Smodel4.rake=np.append(Smodel4.rake,srake[nearestpt])
    Smodel4.slip=np.append(Smodel4.slip,sslip[nearestpt])

#%% load Eric slip

 #k, x(k), y(k), z(k), midLon(k), midLat(k), origin(1), origin(2), lon1(k), lat1(k), lon2(k), lat2(k), ...
 #       upperDepth, lowerDepth, fLength(k), fWidth(k), fDip(k), fStrike(k), rake(k), finalField(k), final_slip_x(k), final_slip_y(k));
  
sinput=np.loadtxt('shakeout_models/pcaim_slip_dist_for_forward_okada.txt')

slon=sinput[:,4]
slat=sinput[:,5]
sdep=sinput[:,3]
#srake=sinput[:,18]
sslip=sinput[:,19]

#for each patch in the fault model, find the nearest input slip patch and assign its rake and slip.
#Use zero if the patch is more than 20km away from any comcot patch

for i in range(Fmodel.npatches):
    distarray=np.array([])
    for j in range(len(slat)):
        distarray=np.append(distarray,np.sqrt((Fmodel.depth[i] - sdep[j])**2 + geod_transform.haversine(slat[j],slon[j],Fmodel.latc[i],Fmodel.lonc[i])**2))
    #if((Fmodel.depth[i]>20000 and min(distarray)>20000) or min(distarray)>50000):
    #    Smodel.slip=np.append(Smodel.slip,0)
    #else:
    nearestpt=np.where(distarray==min(distarray))[0][0]
    #Smodel5.rake=np.append(Smodel5.rake,srake[nearestpt])
    Smodel5.slip=np.append(Smodel5.slip,sslip[nearestpt])



#%% load QQ slip - ENIF

sinput=np.loadtxt('shakeout_models/Mentawai_shakeout_slip_ENIFCorrect_reformat.txt')
slon=sinput[:,0]
slat=sinput[:,1]
sdep=sinput[:,2]
#sstr=sinput[:,3]
#sdip=sinput[:,4]
#slen=sinput[:,5]
#swid=sinput[:,6]
sslip=sinput[:,7]

#for each patch in the fault model, find the nearest input slip patch and assign its rake and slip.
#Use zero if the patch is more than 20km away from any comcot patch

for i in range(Fmodel.npatches):
    distarray=np.array([])
    for j in range(len(slat)):
        distarray=np.append(distarray,np.sqrt((Fmodel.depth[i] - sdep[j])**2 + geod_transform.haversine(slat[j],slon[j],Fmodel.latc[i],Fmodel.lonc[i])**2))
    #if((Fmodel.depth[i]>20000 and min(distarray)>20000) or min(distarray)>50000):
    #    Smodel.slip=np.append(Smodel.slip,0)
    #else:
    nearestpt=np.where(distarray==min(distarray))[0][0]
    #Smodel6.rake=np.append(Smodel6.rake,srake[nearestpt])
    Smodel6.slip=np.append(Smodel6.slip,sslip[nearestpt])






  


#%%


# 2D plotting commands.


#create the plot
Fplot=fault_plots.FaultPlot2D()

#plot gps sites
sugarsites=np.loadtxt('shakeout_data/SuGAR.sites',usecols=(1,2))
sugarlon=sugarsites[:,0]
sugarlat=sugarsites[:,1]
#sugarelev=0*sugarlat

sumosites=np.loadtxt('shakeout_data/SuMo.sites',usecols=(1,2))
sumolon=sumosites[:,0]
sumolat=sumosites[:,1]
#sumoelev=0*sumolat

Fplot.plot_symbols(sugarlon,sugarlat,'ob')
Fplot.plot_symbols(sumolon,sumolat,'or')

#plot epicenter
#Fplot.plot_symbols([hypolon,hypolon],[hypolat,hypolat],'w*',markersize=15,alpha=0.5)

#plot trench axis
trench=np.loadtxt('shakeout_data/sunda_trench.xy',comments='>')
trenchlon=trench[:,0]
trenchlat=trench[:,1]
#trenchelev=trench[:,2]
Fplot.plot_outlines(trenchlon,trenchlat,'--k')

#plot sumatran fault
sf=np.loadtxt('shakeout_data/great_sumatra_fault_nan.xy')
sflon=sf[:,0]
sflat=sf[:,1]
#sfelev=0*sflon
Fplot.plot_outlines(sflon,sflat,'r')

Fplot.plot_shapefile('shakeout_data/sumatra')
plt.set_cmap('RdBu')
#Fplot.plot_slip_patches(Fmodel, Smodel5.slip/100, clim=[0,10])
Fplot.plot_slip_patches(Fmodel, Smodel.slip - Smodel3.slip, clim=[-5,5])


#Fplot.plot_slip_patches(Fmodel, Smodel.slip - Smodel4.slip, clim=[-5,5])
#Fplot.plot_slip_patches(Fmodel, Smodel.slip - Smodel3.slip, clim=[-5,5])
#Fplot.plot_slip_patches(Fmodel,Smodel.slip - Fmodel,Smodel2.slip/100,clim=[-6,6])


Fplot.set_lims([95,105],[-5,1])
Fplot.showmap()


#%%

import matplotlib.pyplot as plt
import shapefile 
from matplotlib import collections

#load data
trench=np.loadtxt('shakeout_data/sunda_trench.xy',comments='>')
trenchlon=trench[:,0]
trenchlat=trench[:,1]
sf=np.loadtxt('shakeout_data/great_sumatra_fault_nan.xy')
sflon=sf[:,0]
sflat=sf[:,1]

coast = shapefile.Reader('shakeout_data/sumatra')

#create the plot
fig = plt.figure()

#first subplot
ax = fig.add_subplot(2,3,1)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude') 
ax.set_title('Input Model') 
ax.axis('equal')
ax.plot(trenchlon,trenchlat,'--k')
ax.plot(sflon,sflat,'r')
ax.plot(sugarlon,sugarlat,'og',markersize=4)
for shape in coast.shapes():   
    x, y = zip(*shape.points)
    ax.plot(x,y,color='k')
plt.set_cmap('jet')
verts=Fmodel.get_patch_verts_center_2d()
poly=collections.PolyCollection(verts,linewidths=0)
colset = np.array(Smodel.slip)
poly.set_array(colset)
poly.set_clim(0,10)
i1=ax.add_collection(poly)
ax.set_xlim(97,102.5)
ax.set_ylim(-5,0)


#second subplot
rms=np.sqrt(np.mean((Smodel.slip-Smodel2.slip/100)**2))

ax = fig.add_subplot(2,3,2)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude') 
ax.set_title('Inverse 2 - QQ: rms %.2f'%rms) 
ax.axis('equal')
ax.plot(trenchlon,trenchlat,'--k')
ax.plot(sflon,sflat,'r')
ax.plot(sugarlon,sugarlat,'og',markersize=4)
for shape in coast.shapes():   
    x, y = zip(*shape.points)
    ax.plot(x,y,color='k')
plt.set_cmap('RdBu')
verts=Fmodel.get_patch_verts_center_2d()
poly=collections.PolyCollection(verts,linewidths=0)
colset = np.array(Smodel.slip-Smodel2.slip/100)
poly.set_array(colset)
poly.set_clim(-5,5)
ax.add_collection(poly)
ax.set_xlim(97,102.5)
ax.set_ylim(-5,0)


#third subplot
rms=np.sqrt(np.mean((Smodel.slip-Smodel3.slip)**2))

ax = fig.add_subplot(2,3,3)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude') 
ax.set_title('GTDef - Lujia: rms %.2f'%rms) 
ax.axis('equal')
ax.plot(trenchlon,trenchlat,'--k')
ax.plot(sflon,sflat,'r')
ax.plot(sugarlon,sugarlat,'og',markersize=4)
for shape in coast.shapes():   
    x, y = zip(*shape.points)
    ax.plot(x,y,color='k')
plt.set_cmap('RdBu')
verts=Fmodel.get_patch_verts_center_2d()
poly=collections.PolyCollection(verts,linewidths=0)
colset = np.array(Smodel.slip-Smodel3.slip)
poly.set_array(colset)
poly.set_clim(-5,5)
ax.add_collection(poly)
ax.set_xlim(97,102.5)
ax.set_ylim(-5,0)

#fourth subplot
rms=np.sqrt(np.mean((Smodel.slip-Smodel4.slip)**2))

ax = fig.add_subplot(2,3,4)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude') 
ax.set_title('PCAIM - Louisa: rms %.2f'%rms) 
ax.axis('equal')
ax.plot(trenchlon,trenchlat,'--k')
ax.plot(sflon,sflat,'r')
ax.plot(sugarlon,sugarlat,'og',markersize=4)
for shape in coast.shapes():   
    x, y = zip(*shape.points)
    ax.plot(x,y,color='k')
plt.set_cmap('RdBu')
verts=Fmodel.get_patch_verts_center_2d()
poly=collections.PolyCollection(verts,linewidths=0)
colset = np.array(Smodel.slip-Smodel4.slip)
poly.set_array(colset)
poly.set_clim(-5,5)
ax.add_collection(poly)
ax.set_xlim(97,102.5)
ax.set_ylim(-5,0)

#fifth subplot
rms=np.sqrt(np.mean((Smodel.slip-Smodel5.slip/100)**2))

ax = fig.add_subplot(2,3,5)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude') 
ax.set_title('PCAIM - Eric: rms %.2f'%rms) 
ax.axis('equal')
ax.plot(trenchlon,trenchlat,'--k')
ax.plot(sflon,sflat,'r')
ax.plot(sugarlon,sugarlat,'og',markersize=4)
for shape in coast.shapes():   
    x, y = zip(*shape.points)
    ax.plot(x,y,color='k')
plt.set_cmap('RdBu')
verts=Fmodel.get_patch_verts_center_2d()
poly=collections.PolyCollection(verts,linewidths=0)
colset = np.array(Smodel.slip-Smodel5.slip/100)
poly.set_array(colset)
poly.set_clim(-5,5)
i5=ax.add_collection(poly)
ax.set_xlim(97,102.5)
ax.set_ylim(-5,0)



#sixth subplot
rms=np.sqrt(np.mean((Smodel.slip-Smodel6.slip)**2))

ax = fig.add_subplot(2,3,6)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude') 
ax.set_title('ENIF - QQ: rms %.2f'%rms) 
ax.axis('equal')
ax.plot(trenchlon,trenchlat,'--k')
ax.plot(sflon,sflat,'r')
ax.plot(sugarlon,sugarlat,'og',markersize=4)
for shape in coast.shapes():   
    x, y = zip(*shape.points)
    ax.plot(x,y,color='k')
plt.set_cmap('RdBu')
verts=Fmodel.get_patch_verts_center_2d()
poly=collections.PolyCollection(verts,linewidths=0)
colset = np.array(Smodel.slip-Smodel6.slip)
poly.set_array(colset)
poly.set_clim(-5,5)
i6=ax.add_collection(poly)
ax.set_xlim(97,102.5)
ax.set_ylim(-5,0)

colorbar_ax = fig.add_axes([0.735, 0.1, 0.015, 0.16])
cb=fig.colorbar(i6, cax=colorbar_ax)
cb.set_ticks([-5,0,5], update_ticks=True)

colorbar_ax = fig.add_axes([0.08, 0.6, 0.015, 0.16])
cb=fig.colorbar(i1, cax=colorbar_ax)
cb.set_ticks([0,5,10], update_ticks=True)

plt.tight_layout(pad=0.4, w_pad=0.0, h_pad=0.2)

fig.show()

