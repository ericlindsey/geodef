# Eric Lindsey, 04/2016

import fault_model
import slip_model
import fault_plots
import numpy as np
import geod_transform
import os.path
import matplotlib.pyplot as plt


Fmodel=fault_model.FaultModel()
Smodel=slip_model.SlipModel()
Smodel.set_fault_model(Fmodel)

fname='shakeout_data/sf_slipmodel.txt'

gpsdata=('shakeout_data/SuGAr.sites','shakeout_data/SuMo.sites')
insardata=('shakeout_data/p137_comb.llenu','shakeout_data/p35_f3700.llenu')

latcorner=-2.76
loncorner=101.93
depthcorner=0
strike=138
dip=77
L=100000
W=18000
nL=50
nW=12

# slip timing
hypolat=-3.076
hypolon=102.199
hypodepth=9030
rupvel=2660

rake=186
risetime=6
    

#recreate0
#if recreate==1:
#    #create fault model
#    Fmodel.create_planar_model(latcorner,loncorner,depthcorner,strike,dip,L,W,nL,nW)
#
#    #create synthetic slip
#    Smodel.random_slip([1,0,0,0],7.1,rake,risetime,hypolon,hypolat,hypodepth,rupvel)
#    
#    #customize slip values
#    ID=np.where(Smodel.F.dipid==2)
#    Smodel.slip[ID]=2
#    ID=np.where(Smodel.F.dipid==4)
#    Smodel.slip[ID]=2
#    ID=np.where((Smodel.F.dipid==4) & (Smodel.F.strikeid>11) & (Smodel.F.strikeid<26))
#    Smodel.slip[ID]=3
#    ID=np.where((Smodel.F.dipid==3) & (Smodel.F.strikeid>4) & (Smodel.F.strikeid<12))
#    Smodel.slip[ID]=3
#    ID=np.where((Smodel.F.strikeid>30) & (Smodel.F.strikeid<42))
#    Smodel.slip[ID]=2
#    ID=np.where(Smodel.F.depth>11000)
#    Smodel.slip[ID]=0
#    
#    #filter the slip
#    Smodel.smooth_slip([1,0,0,0],7.1,2)
#    
#    Smodel.save_slip(fname)
#    
#else:
Smodel.load_slip_static(fname)
    
#compute greens functions and displacements - GPS
redo_gps=0
if(redo_gps==1):
    for gpsfname in gpsdata:
        #load gps site locations and names
        gps=np.loadtxt(gpsfname,usecols=(1,2,3)) # format is any but LON LAT must be columns 1,2
        gpsnames=np.genfromtxt(gpsfname,usecols=(0,),dtype=np.str)
        #get displacements
        gpseast,gpsnorth,gpsup=Smodel.forward_model_static(gps[:,1],gps[:,0])
        gpsmag=np.sqrt(gpseast**2 + gpsnorth**2 + gpsup**2)
        Imoved=np.where(gpsmag>0.003)
        #synthetic errors
        horizsig=1.0
        vertsig=3.0
        gpseast=1000*gpseast + horizsig*np.random.standard_normal(np.shape(gpseast))
        gpsnorth=1000*gpsnorth + horizsig*np.random.standard_normal(np.shape(gpseast))
        gpsup=1000*gpsup + horizsig*np.random.standard_normal(np.shape(gpseast))
        
        gpserre=np.repeat(horizsig,np.shape(gpseast))
        gpserrn=np.repeat(horizsig,np.shape(gpseast))
        gpserru=np.repeat(vertsig,np.shape(gpseast))
        weight=np.repeat(1.0,np.shape(gpseast))
        t0=np.repeat('20160427',np.shape(gpseast))
        t0d=np.repeat('2016.32030',np.shape(gpseast))
        t1=np.repeat('20160427',np.shape(gpseast))
        t1d=np.repeat('2016.32030',np.shape(gpseast))
        cne=np.repeat(0.0,np.shape(gpseast))
        #output file name and header
        gps_outfname=os.path.splitext(gpsfname)[0] + '_SF.vel'
        gpsheader='EQ coseismic deformation extracted from <synthetic_model>\n1    2   3   4    5   6    7    8    9          10    11  12\nYear Mon Day Hour Min Sec  Lon  Lat  Depth[km]  Mag   ID  Decyr\n2016 04 27 07 21 23.1400  102.19900   -3.07600  9.3000  7.10    201604270001     2016.320301\n1   2   3   4      5  6  7  8   9   10  11      12 13  14 15    16\nSta Lon Lat Height OE ON OU ErE ErN ErU Weight  T0 T0D T1 T1D   Cne\nHeight [m] Disp [mm] error [mm]'
        # output format: Sta Lon Lat Height UE UN UU ErE ErN ErU Weight  T0 T0D T1 T1D   Cne
        gps_out=np.column_stack((gpsnames[Imoved],gps[Imoved,0].flatten(),gps[Imoved,1].flatten(),gps[Imoved,2].flatten(),np.round(gpseast[Imoved],1),np.round(gpsnorth[Imoved],1),np.round(gpsup[Imoved],1),gpserre[Imoved],gpserrn[Imoved],gpserru[Imoved],weight[Imoved],t0[Imoved],t0d[Imoved],t1[Imoved],t1d[Imoved],cne[Imoved]))      
        print(gpsnames)
        np.savetxt(gps_outfname,gps_out,fmt='%s',header=gpsheader)
        
        plt.figure()
        ax = plt.gca()
        ax.quiver(gps[:,0],gps[:,1],gpseast,gpsnorth,angles='xy',scale_units='xy')
        plt.draw()
        plt.show()

#compute greens functions and displacements - InSAR

redo_insar=0
if(redo_insar==1):
    for insarfname in insardata:
        insar=np.loadtxt(insarfname) # format LON LAT LOSE LOSN LOSU
        ieast,inorth,iup=Smodel.forward_model_static(insar[:,1],insar[:,0]) # inputs are Lat, Lon
        ilos=ieast*insar[:,2] + inorth*insar[:,3] + iup*insar[:,4] # East + North + Up
        isig=-1+0*ilos #create fake sigmas to match the file format  
        insar_outfname=os.path.splitext(insarfname)[0] + '_SF.lltenuds'
        # output format LON LAT ELEV LOSE LOSN LOSU LOSDISP SIGMA
        insar_out=np.column_stack((insar[:,0],insar[:,1],0*insar[:,1],insar[:,2],insar[:,3],insar[:,4],np.round(1000*ilos,1),isig))
        np.savetxt(insar_outfname,insar_out,fmt='%.9f %.9f %.2f %.5f %.5f %.5f %.1f %.1f')

        plt.figure()
        ax = plt.gca()
        ax.scatter(insar[:,0],insar[:,1],c=ilos,s=20,lw = 0)
        plt.draw()
        plt.show()
        
#%%
insar=np.loadtxt(insarfname) # format LON LAT LOSE LOSN LOSU
#ieast,inorth,iup=Smodel.forward_model_static(insar[:,1],insar[:,0]) # inputs are Lat, Lon
ilos=ieast*insar[:,2] + inorth*insar[:,3] + iup*insar[:,4] # East + North + Up
isig=-1+0*ilos #create fake sigmas to match the file format  
insar_outfname=os.path.splitext(insarfname)[0] + '.lltenuds'
# output format LON LAT ELEV LOSE LOSN LOSU LOSDISP SIGMA
insar_out=np.column_stack((insar[:,0],insar[:,1],0*insar[:,1],insar[:,2],insar[:,3],insar[:,4],np.round(1000*ilos,1),isig))
np.savetxt(insar_outfname,insar_out,fmt='%.9f %.9f %.2f %.5f %.5f %.5f %.1f %.1f')

plt.figure()
ax = plt.gca()
ax.scatter(insar[:,0],insar[:,1],c=ilos,s=20,lw = 0)

plt.draw()
plt.show()

#%%


#plotting commands.
#create the plot
Fplot=fault_plots.FaultPlot3D()
#plot gps sites
sugarsites=np.loadtxt('shakeout_data/SuGAR.sites',usecols=(1,2))
sugarlon=sugarsites[:,0]
sugarlat=sugarsites[:,1]
sugarelev=0*sugarlat
sumosites=np.loadtxt('shakeout_data/SuMo.sites',usecols=(1,2))
sumolon=sumosites[:,0]
sumolat=sumosites[:,1]
sumoelev=0*sumolat
Fplot.plot_symbols(sugarlon,sugarlat,sugarelev,'ob')
Fplot.plot_symbols(sumolon,sumolat,sumoelev,'or')

#plot trench axis
trench=np.loadtxt('shakeout_data/sunda_trench.xy',comments='>')
trenchlon=trench[:,0]
trenchlat=trench[:,1]
trenchelev=trench[:,2]
Fplot.plot_outlines(trenchlon,trenchlat,trenchelev,'--k')

#plot epicenter
Fplot.plot_symbols([hypolon,hypolon],[hypolat,hypolat],[1.e-3*hypodepth,1.e-3*hypodepth],'w*',markersize=15,alpha=0.5)

#plot sumatran fault
sf=np.loadtxt('shakeout_data/great_sumatra_fault_nan.xy')
sflon=sf[:,0]
sflat=sf[:,1]
sfelev=0*sflon
Fplot.plot_outlines(sflon,sflat,sfelev,'r')

Fplot.plot_shapefile('shakeout_data/sumatra')

Fplot.plot_slip_patches(Smodel.F,Smodel.slip)

#Fplot.plot_vectors(sugarlon,sugarlat,sugarelev, sugareast,sugarnorth,sugarup, length=1)

Fplot.set_lims([101,103],[-4,-1],[-4,200])
Fplot.showmap()

#%%

#save moment tensor text files

import moment_tensor


#hypocenter parameters
strike=138
dip=77
rake=186

hypolat=-3.076
hypolon=102.199
hypodepth=9030


# slip timing
rupvel=2660 # m/sec
patchduration=6.

year=2016
month=4
day=26
hour=7
minute=21
sec=40.0

for i in range(Fmodel.npatches):
    if(Smodel.slip[i]>0):
        patchid='simulated%d%02d%02d_%04d'%(year,month,day,i)
        patchdist= np.sqrt((Fmodel.depth[i]-hypodepth)**2 + geod_transform.haversine(hypolat,hypolon,Fmodel.latc[i],Fmodel.lonc[i])**2)
        patchdelay=patchdist/rupvel
        patchsec=np.mod(patchdelay+sec,60)
        patchmin=minute+np.floor_divide(patchdelay+sec,60)
        
        fname='shakeout_data/sf_mt_all_notimeshift/%s.txt'%patchid
        
        moment_tensor.save_moment_tensor(fname,Fmodel.strike[i],Fmodel.dip[i],rake,Fmodel.L[i],Fmodel.W[i],Smodel.slip[i],Fmodel.lonc[i],Fmodel.latc[i],Fmodel.depth[i]*1.e-3,year,month,day,hour,patchmin,patchsec,patchduration,patchid)
        

#for i in range(Fmodel.npatches):
#    if(Smodel.slip[i]>0):
#        patchid='simulated%d%02d%02d_%04d'%(year,month,day,i)
#        patchmoment=1e7*30e9*Fmodel.L[i]*Fmodel.W[i]*Smodel.slip[i]
#        patchmagnitude=(2/3)*np.log10(patchmoment) - 10.7
#        patchdist= np.sqrt((Fmodel.depth[i]-hypodepth)**2 + geod_transform.haversine(hypolat,hypolon,Fmodel.latc[i],Fmodel.lonc[i])**2)
#        patchdelay=patchdist/rupvel
#        patchsec=np.mod(patchdelay+sec,60)
#        patchmin=minute+np.floor_divide(patchdelay+sec,60)
#        Mrr,Mtt,Mpp,Mrt,Mrp,Mtp = moment_tensor.get_moment_tensor(Fmodel.strike[i],Fmodel.dip[i],rake,patchmoment)
#        if(patchdelay>maxdelay):
#            maxdelay=patchdelay
#        fname='shakeout_data/sf_mt_all_notimeshift/%s.txt'%patchid
#        f1=open(fname, 'w')
#        f1.write("%d %02d %02d %02d %02d %05.2f %.4f %.4f %.4f %.1f %.1f %s\n"%(year,month,day,hour,patchmin,patchsec,Fmodel.latc[i],Fmodel.lonc[i],Fmodel.depth[i]*1.e-3,patchmagnitude,patchmagnitude,patchid))
#        f1.write("event name:    %s\n"%patchid)
#        f1.write("time shift:    0.0000\n")
#        f1.write("half duration: %.4f\n"%(patchduration/2.))
#        f1.write("latitude:      %.4f\n"%Fmodel.latc[i])
#        f1.write("longitude:     %.4f\n"%Fmodel.lonc[i])
#        f1.write("depth:         %.4f\n"%Fmodel.depth[i])
#        f1.write("Mrr:           %.6e\n"%Mrr)
#        f1.write("Mtt:           %.6e\n"%Mtt)
#        f1.write("Mpp:           %.6e\n"%Mpp)
#        f1.write("Mrt:           %.6e\n"%Mrt)
#        f1.write("Mrp:           %.6e\n"%Mrp)
#        f1.write("Mtp:           %.6e\n"%Mtp)
#        f1.write("\n")
#        f1.close()
#print(maxdelay)
#
##%%