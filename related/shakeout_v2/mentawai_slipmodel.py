# Eric Lindsey, 04/2016

# todo: tuesday
# generate InSAR for Mentawai
# make plots and double check data


import numpy as np
import fault_model
import fault_plots
import geod_transform
import slip_model
import os
import matplotlib.pyplot as plt

Fmodel=fault_model.FaultModel()
Smodel=slip_model.SlipModel()
Smodel.set_fault_model(Fmodel)

fname='shakeout_data/mentawai_10km2_3d_slipmodel.txt'

hypolon=98.9809
hypolat=-1.0386
hypodepth=30277 #in meters

rake=97
rupvel=2660 # m/sec
patchduration=15.
#strike=322.7
#dip=15.0

year=2016
month=4
day=27
hour=4
minute=47
sec=15.5

gpsdata=('shakeout_data/SuGAr.sites','shakeout_data/SuMo.sites')
insardata=('shakeout_data/p36_f3650.llenu','shakeout_data/p35_f3700.llenu','shakeout_data/p37_f3650.llenu')

recreate=0
if recreate==1:
    #load fault model

    Fmodel.load_patches_topleft('shakeout_data/mentawai_10km2_3d_subfaults.out')

    #create synthetic slip
    comcot=np.loadtxt('shakeout_data/fault_data_2b_comcot_1_7_edit.txt')
    comcotslip=comcot[:,9]
    comcotlon=comcot[:,1]
    comcotlat=comcot[:,2]
    #for each patch in the fault model, find the nearest comcot slip patch and assign its slip.
    #Use zero if the patch is more than 20km away from any comcot patch
    Smodel.slip=np.array([])
    for i in range(Fmodel.npatches):
        distarray=np.array([])
        for j in range(len(comcotlat)):
            distarray=np.append(distarray,geod_transform.haversine(comcotlat[j],comcotlon[j],Fmodel.latc[i],Fmodel.lonc[i]))
        if((Fmodel.depth[i]>20000 and min(distarray)>20000) or min(distarray)>50000):
            Smodel.slip=np.append(Smodel.slip,0)
        else:
            nearestpt=np.where(distarray==min(distarray))[0][0]
            Smodel.slip=np.append(Smodel.slip,comcotslip[nearestpt])
    
    # rescale magnitude to 8.9
    # magnitude=(2/3)*log10(moment_in_dyne_cm) - 10.7
    # moment_in_dyne_cm=10^((3/2)*(mangitude+10.7))
    mo=1e7*30e9*sum(np.multiply(np.multiply(Smodel.slip,Fmodel.L),Fmodel.W))
    mo_desired=10**((3/2)*(8.9+10.7))
    rescale=mo_desired/mo
    Smodel.slip=Smodel.slip*rescale
    mo_new=1e7*30e9*sum(np.multiply(np.multiply(Smodel.slip,Fmodel.L),Fmodel.W))
    magnitude=(2/3)*np.log10(mo_new) - 10.7
    
    print("simulated magnitude ", magnitude)
    
    Smodel.risetime = patchduration * np.ones(Smodel.F.npatches)
    Smodel.rake = rake * np.ones(Smodel.F.npatches)
    Smodel.onsettime = Smodel.slip_time(hypolat,hypolon,hypodepth,rupvel)

    Smodel.save_slip(fname)
    
else: 
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
        gps_outfname=os.path.splitext(gpsfname)[0] + '_Mentawai.vel'
        gpsheader='EQ coseismic deformation extracted from <synthetic_model>\n1    2   3   4    5   6    7    8    9          10    11  12\nYear Mon Day Hour Min Sec  Lon  Lat  Depth[km]  Mag   ID  Decyr\n2016 04 27 04 47 15.5000  98.9809   -1.0386  30.28  8.90    201604270000     2016.320128\n1   2   3   4      5  6  7  8   9   10  11      12 13  14 15    16\nSta Lon Lat Height OE ON OU ErE ErN ErU Weight  T0 T0D T1 T1D   Cne\nHeight [m] Disp [mm] error [mm]'
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

#insardata=('shakeout_data/p37_f3650.llenu','shakeout_data/p35_f3700.llenu')

redo_insar=0
if(redo_insar==1):
    for insarfname in insardata:
        insar=np.loadtxt(insarfname) # format LON LAT LOSE LOSN LOSU
        ieast,inorth,iup=Smodel.forward_model_static(insar[:,1],insar[:,0]) # inputs are Lat, Lon
        ilos=ieast*insar[:,2] + inorth*insar[:,3] + iup*insar[:,4] # East + North + Up
        isig=-1+0*ilos #create fake sigmas to match the file format  
        insar_outfname=os.path.splitext(insarfname)[0] + '_Mentawai.lltenuds'
        # output format LON LAT ELEV LOSE LOSN LOSU LOSDISP SIGMA
        insar_out=np.column_stack((insar[:,0],insar[:,1],0*insar[:,1],insar[:,2],insar[:,3],insar[:,4],np.round(1000*ilos,1),isig))
        np.savetxt(insar_outfname,insar_out,fmt='%.9f %.9f %.2f %.5f %.5f %.5f %.1f %.1f')

        plt.figure()
        ax = plt.gca()
        sc=ax.scatter(insar[:,0],insar[:,1],c=ilos,s=20,lw = 0)
        plt.colorbar(sc)
        plt.draw()
        plt.show()
        
#%%

# 3D plotting commands.

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

#plot epicenter
Fplot.plot_symbols([hypolon,hypolon],[hypolat,hypolat],[1.e-3*hypodepth,1.e-3*hypodepth],'w*',markersize=15,alpha=0.5)

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


#Fplot.plot_vectors(sugarlon,sugarlat,sugarelev, sugareast,sugarnorth,sugarup, length=1)


Fplot.plot_shapefile('shakeout_data/sumatra')
Fplot.plot_slip_patches(Fmodel,0*Smodel.slip)
Fplot.set_lims([98,104],[-4,4],[-4,400])
Fplot.showmap()

#%%

# 2D plotting commands.
import fault_plots


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
Fplot.plot_symbols([hypolon,hypolon],[hypolat,hypolat],'w*',markersize=15,alpha=0.5)

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

Fplot.plot_slip_patches(Fmodel,Smodel.slip)

# load sugar data
gps_fname='shakeout_data/SuGAr_Mentawai.vel'
# format: Sta Lon Lat Height UE UN UU ErE ErN ErU Weight  T0 T0D T1 T1D   Cne
sugar_disp=np.loadtxt(gps_fname,usecols=(1,2,4,5,6))
sugarlon=sugar_disp[:,0]
sugarlat=sugar_disp[:,1]
sugareast=sugar_disp[:,2]
sugarnorth=sugar_disp[:,3]
sugarup=sugar_disp[:,4]
#Fplot.plot_vectors(sugarlon,sugarlat, sugareast,sugarnorth,scale=2000)
#Fplot.plot_up_vectors(sugarlon,sugarlat, sugarup ,scale=2000)

Fplot.set_lims([95,105],[-5,1])
Fplot.showmap()


#%%
#
##save moment tensor text files
#
import moment_tensor

hypolon=98.9809
hypolat=-1.0386
hypodepth=30277 #in meters

rake=97
rupvel=2660 # m/sec
patchduration=15.

year=2016
month=4
day=25
hour=0
minute=47
sec=15.0

for i in range(Fmodel.npatches):
    if(Smodel.slip[i]>0):
        patchid='simulated%d%02d%02d_%04d'%(year,month,day,i)
        patchdist= np.sqrt((Fmodel.depth[i]-hypodepth)**2 + geod_transform.haversine(hypolat,hypolon,Fmodel.latc[i],Fmodel.lonc[i])**2)
        patchdelay=patchdist/rupvel
        patchsec=np.mod(patchdelay+sec,60)
        patchmin=minute+np.floor_divide(patchdelay+sec,60)
        
        fname='shakeout_data/mentawai_mt_all_notimeshift/%s.txt'%patchid
        
        moment_tensor.save_moment_tensor(fname,Fmodel.strike[i],Fmodel.dip[i],rake,Fmodel.L[i],Fmodel.W[i],Smodel.slip[i],Fmodel.lonc[i],Fmodel.latc[i],Fmodel.depth[i]*1.e-3,year,month,day,hour,patchmin,patchsec,patchduration,patchid)
        



#
#for i in range(Fmodel.npatches):
#    if(slip[i]>0):
#        patchid='simulated%d%02d%02d_%04d'%(year,month,day,i)
#        patchmoment=1e7*30e9*Fmodel.L[i]*Fmodel.W[i]*slip[i]
#        patchmagnitude=(2/3)*np.log10(patchmoment) - 10.7
#        patchdist= np.sqrt((Fmodel.depth[i]-hypodepth)**2 + geod_transform.haversine(hypolat,hypolon,Fmodel.latc[i],Fmodel.lonc[i])**2)
#        patchdelay=patchdist/rupvel
#        patchsec=np.mod(patchdelay+sec,60)
#        patchmin=minute+np.floor_divide(patchdelay+sec,60)
#        Mrr,Mtt,Mpp,Mrt,Mrp,Mtp = moment_tensor.get_moment_tensor(Fmodel.strike[i],Fmodel.dip[i],rake,patchmoment)
#        fname='shakeout_data/mentawai_mt_all/%s.txt'%patchid
#        f1=open(fname, 'w')
#        f1.write("%d %02d %02d %02d %02d %05.2f %.4f %.4f %.4f %.1f %.1f %s\n"%(year,month,day,hour,minute,sec,Fmodel.latc[i],Fmodel.lonc[i],Fmodel.depth[i]*1.e-3,patchmagnitude,patchmagnitude,patchid))
#        f1.write("event name:    %s\n"%patchid)
#        f1.write("time shift:    %.4f\n"%patchdelay)
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
