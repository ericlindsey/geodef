#!/usr/local/bin/python
# Eric Lindsey, 01/2014

import numpy as np
import geod_transform
import okada_greens
import fault_model

class SlipModel:
    
    def __init__(self):
        self.F = None
        self.slip = np.array([])
        self.rake = np.array([])
        self.risetime = np.array([])
        self.onsettime = np.array([])
        
    def set_fault_model(self,F):
        self.F = F
        
    def random_slip(self,bound_cond,magnitude,rake,risetime,hypolon,hypolat,hypodepth,rupvel):
        self.slip=np.random.rand(self.F.npatches)
        #set boundary conditions
        self.set_bc_slip(bound_cond)    
        #rescale to desired magnitude
        self.scale_to_magnitude(magnitude)
        #constant rake, other parameters
        self.rake=rake*np.ones(self.F.npatches)
        self.risetime=risetime*np.ones(self.F.npatches)
        self.onsettime=self.slip_time(hypolat,hypolon,hypodepth,rupvel)
        
    def smooth_slip(self,bound_cond,magnitude,smoothing_iter):
        for k in range(smoothing_iter):
            for i in range(self.F.npatches):
                rti=self.F.find_patch(i,1,0)
                lti=self.F.find_patch(i,-1,0) 
                upi=self.F.find_patch(i,0,-1)
                dni=self.F.find_patch(i,0,1)
                self.slip[i]=(2*self.slip[i]+self.slip[rti]+self.slip[lti]+self.slip[upi]+self.slip[dni])/6
            #slip is positive
            self.slip=self.slip - min(self.slip)      
            #set boundary conditions
            self.set_bc_slip(bound_cond)   
        #finally, rescale to desired magnitude
        self.scale_to_magnitude(magnitude)
        
    def load_slip_timedep(self,fname):
        #load slip (and fault model)
        self.F.load_patches_center(fname)
        
        indata=np.loadtxt(fname,ndmin=2)
        self.rake=indata[:,10]
        self.slip=indata[:,11]
        self.risetime=indata[:,12]
        self.onsettime=indata[:,13]
        
    def load_slip_static(self,fname):
        #load slip (and fault model)
        self.F.load_patches_center(fname)
        indata=np.loadtxt(fname,ndmin=2)
        self.rake=indata[:,10]
        self.slip=indata[:,11]
        
    def save_slip(self,fname):
        #save slip (and fault model)
        outdata=np.column_stack((range(self.F.npatches),self.F.dipid,self.F.strikeid,
                                 self.F.lonc,self.F.latc,self.F.depth,self.F.L,self.F.W,
                                 self.F.strike,self.F.dip,self.rake,self.slip,
                                 self.risetime,self.onsettime))
        np.savetxt(fname,outdata,fmt='%10.5f')
        
            
    def set_bc_slip(self,bound_cond):
        if(bound_cond[0]==0):
            zerorow=np.where(self.F.dipid==min(self.F.dipid))
            self.slip[zerorow]=0
        if(bound_cond[1]==0):
            zerorow=np.where(self.F.strikeid==max(self.F.strikeid))
            self.slip[zerorow]=0
        if(bound_cond[2]==0):
            zerorow=np.where(self.F.dipid==max(self.F.dipid))
            self.slip[zerorow]=0
        if(bound_cond[3]==0):
            zerorow=np.where(self.F.strikeid==min(self.F.strikeid))
            self.slip[zerorow]=0
        
    def scale_to_magnitude(self,magnitude):
        rescale=self.get_moment_from_mag(magnitude)/self.get_moment()
        self.slip = rescale * self.slip  
            
    def get_magnitude(self):
        mo=self.get_moment()
        mag=(2/3)*np.log10(mo) - 10.7
        return mag
        
    def get_moment(self):
        mo=1e7*30e9*sum(np.multiply(np.multiply(self.slip,self.F.L),self.F.W))
        return mo
        
    def get_moment_from_mag(self,mag):
        mo=10**((3/2)*(mag+10.7))
        return mo
        
    def slip_time(self,hypolat,hypolon,hypodepth,rupvel):
        patchdist= np.sqrt((self.F.depth-hypodepth)**2 + geod_transform.haversine(hypolat,hypolon,self.F.latc,self.F.lonc)**2)
        patchdelay=patchdist/rupvel
        return patchdelay

    def forward_model_static(self,latobs,lonobs):
        #layout of displacement G is:
        #      ...patch1....
        #    . uE_str uE_dip
        # pt1. uN_str uN_dip
        #    . uU_str uU_dip
        G=self.F.get_greens(latobs,lonobs)
        print(np.shape(G))
        
        edisp=np.array([])
        ndisp=np.array([])
        udisp=np.array([])
        # need to vectorize!
        for i in range(len(latobs)):
            edisp=np.append(edisp,0)
            ndisp=np.append(ndisp,0)
            udisp=np.append(udisp,0)
            for j in range(self.F.npatches):
                sinrake=np.sin(np.radians(self.rake[j]))
                cosrake=np.cos(np.radians(self.rake[j]))
            
                edisp[i]= edisp[i] + self.slip[j]*( cosrake*G[3*i][2*j]   + sinrake*G[3*i][2*j+1]   )
                ndisp[i]= ndisp[i] + self.slip[j]*( cosrake*G[3*i+1][2*j] + sinrake*G[3*i+1][2*j+1] )              
                udisp[i]= udisp[i] + self.slip[j]*( cosrake*G[3*i+2][2*j] + sinrake*G[3*i+2][2*j+1] )
        
        return edisp,ndisp,udisp
        
    def forward_model_dynamic(self,latobs,lonobs):
        # this could be a big project. in general, we'd like to have lots of possible
        # input forward models here, that can be added easily and mixed/matched. 
        return 0
        