# %% import packages
import torch
import numpy as np
import scipy.constants as scc

import sys
import json 
sys.path+=['/research/nfs_ertin_1/nithin_data/mod/blip/python/src/unrolled/util']
from load_bin import nlp_config,radar_config,loadFile,get_cpi_samples,Process_pipelineDecoder,Process_pipeline
propSpeed = scc.c
pi = torch.acos(torch.zeros(1)).item() * 2
from sklearn.preprocessing import normalize
#from scipy.io import savemat

# %% 
class angleDopplerOperator():
    def __init__(self,nlp_opts,Panelarraysize=8,element_spacing=2*0.0254,gpuDev=None ):
        num_az = nlp_opts.num_az
        az_width_rad=nlp_opts.az_width_rad
        num_el = nlp_opts.num_el
        el_width_rad=nlp_opts.el_width_rad
        
        self.gpuDev=gpuDev
        if num_az %2 == 0:
            Azgrid = torch.linspace(-(az_width_rad/2), (az_width_rad/2), 
                                    steps =num_az+1)#,device=self.gpuDev )
            Azgrid = Azgrid[0:num_az]
        else: 
            Azgrid =  torch.linspace( -(az_width_rad/2),(az_width_rad/2), 
                                     steps=num_az)#, device = self.gpuDev )
        if num_el%2 == 0:
            Elgrid =  torch.linspace(-(el_width_rad/2), (el_width_rad/2), 
                                     steps =num_el+1)#, device = self.gpuDev )
            Elgrid = Elgrid[0:num_el]
        else: 
            Elgrid = torch.linspace( -(el_width_rad/2),(el_width_rad/2), 
                                    steps=num_el)#, device = self.gpuDev )   

        self.Azgrid = Azgrid
        self.Elgrid = Elgrid
        self.Panelarraysize = Panelarraysize
        self.element_spacing = element_spacing 
        self.Panelspacing = self.Panelarraysize*self.element_spacing 
        self.subarray_y = torch.tensor([-1 ,0,  1, -2,  2, -3, -1 , 0,  1,  
                                        3, -4, -2 ,2, 4, -3, -1, 0, 1, 3, 
                                        -2 ,2, -1, 0, 1])*self.Panelspacing
        self.subarray_z = torch.tensor([-3, -3, -3, -2, -2, -1, -1, -1, -1, 
                                        -1,  0,  0, 0, 0,  1,  1, 1, 1, 1,  
                                        2, 2,  3, 3, 3])*self.Panelspacing
        self.Subarray_globalweight = torch.tensor([0.622, 0.613, 0.622, 0.867, 
                                                   0.867, 0.618, 0.895, 1, 0.895, 
                                                   0.618, 0.698 ,0.588, 0.588, 
                                                   0.698, 0.618, 0.895, 1, 0.895, 
                                                   0.618, 0.867, 0.867, 0.622, 0.613,
                                                   0.622])
        self.Subarray_count = self.subarray_y.shape[0]
        self.Subarray_yweight = torch.pow(10,torch.tensor([-18,-14,-10,-7,-4,
                                                           -2.5,-1,0,0,-1,-2.5,
                                                           -4,-7,-10,-14,-18])/20).reshape((1,-1))
        self.Subarray_zweight = torch.cat((0.812*torch.ones(self.Panelarraysize,1),
                                           1*torch.ones(self.Panelarraysize,1), 
                                           torch.ones(self.Panelarraysize,1),
                                         0.812*torch.ones(self.Panelarraysize,1)))
        
        
        self.Subarray_weight=torch.matmul(self.Subarray_zweight,self.Subarray_yweight)

        self.Panel_y = torch.tensor([-1.5, -0.5 ,0.5,1.5,-2.5, -1.5, -0.5,  0.5, 1.5, 2.5,-3.5, -2.5, -1.5, -0.5,  
                   0.5,1.5,2.5,3.5,-4.5,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5,-4.5,
                   -3.5, -2.5, -1.5, -0.5,  0.5, 1.5, 2.5, 3.5, 4.5, -4.5, -3.5, -2.5, 
                   -1.5, -0.5 ,0.5, 1.5, 2.5, 3.5, 4.5, -4.5,-3.5,-2.5, -1.5, -0.5,  0.5, 1.5,
                    2.5, 3.5, 4.5,-3.5, -2.5, -1.5, -0.5,  0.5, 1.5, 2.5, 3.5,-2.5, -1.5,
                    -0.5,  0.5, 1.5, 2.5, -1.5, -0.5,  0.5, 1.5])*self.Panelspacing

        self.Panel_z = torch.tensor([-4.5,-4.5,-4.5,-4.5,-3.5, -3.5, -3.5, -3.5, -3.5, -3.5,-2.5,
                        -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5,-1.5, -1.5, -1.5, -1.5,
                        -1.5, -1.5, -1.5, -1.5, -1.5, -1.5,-0.5, -0.5, -0.5, -0.5, -0.5,
                        -0.5, -0.5, -0.5, -0.5, -0.5, 0.5,  0.5,  0.5,  0.5,  0.5, 0.5,
                        0.5,  0.5,  0.5,  0.5, 1.5,  1.5,  1.5,  1.5,  1.5,  1.5 , 1.5,
                        1.5,  1.5,  1.5, 2.5, 2.5,  2.5,  2.5,  2.5,  2.5,  2.5,  2.5,
                        3.5 ,3.5 , 3.5 , 3.5 , 3.5 , 3.5, 4.5,  4.5,  4.5,  4.5])*self.Panelspacing

        self.Panel_count=self.Panel_y.shape[0]
        N=self.Panelarraysize
        ylocal=(torch.linspace(-(N/2-0.5),(N/2-0.5),steps=N)*self.element_spacing).reshape((1,-1))
        ylocal=ylocal.repeat(N,1)
        zlocal= (torch.linspace(-(N/2-0.5),(N/2-0.5),steps=N)*self.element_spacing).reshape([-1,1])
        zlocal=zlocal.repeat(1,N)

        self.Panel_elementcenterY = torch.zeros(self.Panel_count,ylocal.shape[0],ylocal.shape[1])
        self.Panel_elementcenterZ = torch.zeros(self.Panel_count,ylocal.shape[0],ylocal.shape[1])
       
        
        for i in torch.arange(self.Panel_count):
            self.Panel_elementcenterY[i,:,:]=self.Panel_y[i]+ylocal 
            self.Panel_elementcenterZ[i,:,:]=self.Panel_z[i]+zlocal          
        
        self.Subarray_elementY=torch.zeros((self.Subarray_count,
                                            self.Panelarraysize*4,
                                            self.Panelarraysize*2))
        self.Subarray_elementZ=torch.zeros((self.Subarray_count,
                                            self.Panelarraysize*4,
                                            self.Panelarraysize*2))
        
        for i in torch.arange(self.Subarray_count):
            ind=torch.flatten(torch.nonzero((torch.abs(self.Panel_y-self.subarray_y[i])<=self.Panelspacing) & (torch.abs(self.Panel_z-self.subarray_z[i])<=self.Panelspacing*2)))
            xbuf=torch.flatten(torch.permute(torch.flatten(torch.permute(self.Panel_elementcenterY[ind,:,:],(0,2,1)),1),(1,0)))
            ybuf=torch.flatten(torch.permute(torch.flatten(torch.permute(self.Panel_elementcenterZ[ind,:,:],(0,2,1)),1),(1,0)))
            I=torch.argsort(ybuf,stable=True) 
            ys=xbuf[I]
            zs=ybuf[I]
            I2=torch.argsort(ys,stable=True) 
            zs2=zs[I2]
            ys2=ys[I2]
            self.Subarray_elementY[i,:,:]=torch.permute(torch.reshape(ys2,(self.Panelarraysize*2,self.Panelarraysize*4)),(1,0))   
            self.Subarray_elementZ[i,:,:]=torch.permute(torch.reshape(zs2,(self.Panelarraysize*2,self.Panelarraysize*4)),(1,0))  

    

    def arrayFactrATD(self,pos,azbuf,elbuf,w):
        dirvec = torch.cat(((-torch.cos(elbuf)*torch.cos(azbuf)).reshape(1,-1),
                            (-torch.cos(elbuf)*torch.sin(azbuf)).reshape(1,-1),
                            (-torch.sin(elbuf)).reshape(1,-1)))
        tau =  torch.matmul(pos.T,dirvec)
        steer = torch.exp(-1j*2*pi*tau)
        bpsv = steer
        bpat = torch.matmul(torch.reshape(torch.conj_physical(w),(1,-1)),bpsv)
        return bpat
    
    def MITLLSubarrayPattern(self,AZ0,EL0,ElGrid,AzGrid,Nsub,cpi_header):
        fc = cpi_header.CenterFrequency[0]                                            
        wavelengthCenter = propSpeed/fc                
        az = AzGrid
        el = ElGrid 
        azbuf,elbuf = torch.meshgrid(az,el)
 
        
        azbuf = torch.flatten(azbuf).detach()
        elbuf = torch.flatten(elbuf).detach()
        # azb = azbuf.numpy()
        # elb = elbuf.numpy()
        # f1=  '/research/nfs_ertin_1/nithin_data/mod/blip/python/src/unrolled/simulaed_data/mat2.mat'
        
        # sy = self.Subarray_elementY.numpy()
        # sz = self.Subarray_elementZ.numpy()
        # pcy = self.Panel_elementcenterY.numpy()
        # pcz = self.Panel_elementcenterZ.numpy()
        
        # vals_to_save = {'azb':azb,'elb':elb}
        # savemat(f1,vals_to_save)
        
        # f1= '/research/nfs_ertin_1/nithin_data/mod/blip/python/src/unrolled/simulaed_data/mat3.mat'

        # vals_to_save = {'sy':sy,'sz':sz,'pcy':pcy,'pcz':pcz}
        # savemat(f1,vals_to_save)
        
        pos=torch.cat( (0*self.Subarray_elementY[Nsub,:,:].reshape(1,-1),
                        self.Subarray_elementY[Nsub,:,:].reshape(1,-1), 
                        self.Subarray_elementZ[Nsub,:,:].reshape(1,-1)),axis=0)/wavelengthCenter
        dirvec = torch.tensor([-torch.cos(EL0)*torch.cos(AZ0),-torch.cos(EL0)*torch.sin(AZ0),-torch.sin(EL0)])
        tau =  torch.matmul(pos.T,dirvec)
        steer = torch.exp(-1j*2*pi*tau)
        af = self.arrayFactrATD(pos,azbuf,elbuf,steer*self.Subarray_weight.flatten()).reshape(1,-1)
        return af
    
    def velocity_templates_l1( self,cpi_header, nlp_opts ):
        wave_speed_mps = propSpeed * cpi_header.VelocityFactor
	    
        rel_start_times =  torch.tensor(cpi_header.PulseStartTimes).reshape(-1,1)

        vel_axis = torch.linspace( nlp_opts.velocity_min, nlp_opts.velocity_max, 
                                  steps=nlp_opts.num_vel ).reshape(1,-1)
        zero_vel_idx = torch.argmin( torch.abs( vel_axis ) )
        vel_axis = vel_axis - vel_axis[0,zero_vel_idx]
 
        num_pulses =  cpi_header.Dims[1]
        scale_factor = 1.0 / np.sqrt( num_pulses )
        spatial_freq = (2 * pi * torch.tensor(cpi_header.CenterFrequency)/ wave_speed_mps).reshape(-1,1)                                                                
        self.A_vel = scale_factor * torch.tensor(torch.exp( -2j * spatial_freq*rel_start_times *vel_axis ),
                                               dtype=torch.complex64,device=self.gpuDev)    
        
    
    def velocity_templates( self,vel,cpi_header ):
        wave_speed_mps = propSpeed * cpi_header.VelocityFactor
        rel_start_times =  torch.tensor(cpi_header.PulseStartTimes).reshape(-1,1)
        num_pulses =  cpi_header.Dims[1]
        scale_factor = 1.0 / np.sqrt( num_pulses )
        spatial_freq = (2 * pi * torch.tensor(cpi_header.CenterFrequency)/ wave_speed_mps).reshape(-1,1)                                     
        vel_temp = scale_factor * torch.exp( -2j * spatial_freq*rel_start_times *vel )
        return vel_temp
    
    def opInit(self,sensor,nlp_opts,cpi_header):
        AZ0 = torch.tensor(cpi_header.AzElec)
        EL0 = torch.tensor(cpi_header.ElElec)
        Azgrid = self.Azgrid+ AZ0
        Elgrid = self.Elgrid+ EL0
        RxChannelWeights = torch.tensor(sensor.RxChannelWeights)
        
        numRxChannels = RxChannelWeights.shape[0]
        self.A_angle = torch.zeros(numRxChannels,nlp_opts.num_el*nlp_opts.num_az,
                              dtype= torch.complex64,device=self.gpuDev)
        for Nsub in torch.arange(numRxChannels):
            self.A_angle[Nsub,:] = self.MITLLSubarrayPattern(AZ0,EL0,
                                                        Elgrid,Azgrid,Nsub,cpi_header)
          
        self.A_angle = torch.nn.functional.normalize(self.A_angle,p=2,dim=0)

        
        self.velocity_templates_l1( cpi_header, nlp_opts )
        self.A_angle_gramPinv = torch.linalg.pinv(torch.matmul(torch.conj_physical(self.A_angle).T,self.A_angle))
        self.A_vel_gramPinv = torch.linalg.pinv(torch.matmul(torch.conj_physical(self.A_vel).T,self.A_vel)) 
        
    def forward_mat(self,x):
        return torch.matmul(torch.matmul(self.A_vel,x),self.A_angle.T)
    
    def adjoint_mat(self,x):
        return torch.matmul(torch.matmul( torch.conj_physical(self.A_vel).T,x),
                            torch.conj_physical(self.A_angle))
    def M_mat(self,x):
        return torch.matmul(torch.matmul(torch.conj_physical(self.A_vel_gramPinv).T,x),
                            torch.conj_physical(self.A_angle_gramPinv))    
        
    def gram_matrix(self,x):
        return self.adjoint_mat(self.forward_mat(x))

    def least_squares_gram(self,x,y):
        y1 = self.forward_mat(x)
        return self.adjoint_mat(y1-(y))
    
    def lipschitzConstant(self,y,nlp_opts):
        nSamples=1
        x1 = torch.randn(nSamples,2,nlp_opts.num_vel,nlp_opts.num_az*nlp_opts.num_el,
                         dtype=torch.complex64,device=self.gpuDev)
        x2 = torch.randn(nSamples,2,nlp_opts.num_vel,nlp_opts.num_az*nlp_opts.num_el,
                         dtype=torch.complex64,device=self.gpuDev)
        gradf1 = self.least_squares_gram(x1,y)
        gradf2 = self.least_squares_gram(x2,y)
        L=torch.zeros(nSamples)
        for i in torch.arange(nSamples):
            L[i] = torch.norm(gradf1[i,:,:,:].flatten()-gradf2[i,:,:,:].flatten())/torch.norm(x2[i,:,:,:].flatten()-x1[i,:,:,:].flatten())
        Lmax = torch.max(L)
        if Lmax <1e-6:
            Lmax =1e-6
        tau = 2/Lmax/10
        return tau, Lmax    
    
    def generate_response(self,cpi_header,sensor,az,el,vel,amp):
        AZ0 = torch.tensor(cpi_header.AzElec)
        EL0 = torch.tensor(cpi_header.ElElec)
        RxChannelWeights = torch.tensor(sensor.RxChannelWeights)
        
        numRxChannels = RxChannelWeights.shape[0]        
        A_angle = torch.zeros(numRxChannels,1,dtype= torch.complex64)

        for Nsub in torch.arange(numRxChannels):
            A_angle[Nsub,:] = self.MITLLSubarrayPattern(AZ0,EL0,el,az,Nsub,cpi_header)
        A_vel = self.velocity_templates( vel,cpi_header )
        resp = torch.zeros(1,2,A_vel.shape[0],numRxChannels,dtype=torch.complex64)    
        resp[0,0,:,:] = amp[0]*torch.matmul(A_vel,A_angle.T)
        resp[0,1,:,:] = amp[1]*torch.matmul(A_vel,A_angle.T)
        return resp

#%% main     
if __name__ == "__main__":
    fileJson = open('/research/nfs_ertin_1/nithin_data/mod/blip/config/process/process_Static_Tower_09192023.json')
    Process_pipelineObj = json.load(fileJson,object_hook=Process_pipelineDecoder)

    fileJsonradar = open(Process_pipelineObj.radar_config)
    fileJsonnlp = open(Process_pipelineObj.nlp_config)


    nlp_configObjDict = json.load(fileJsonnlp)
    nlp_configObj = nlp_config(nlp_configObjDict['range_min'], nlp_configObjDict['range_max'],
                            nlp_configObjDict['range_oversample'],nlp_configObjDict['velocity_min'],
                            nlp_configObjDict['velocity_max'],nlp_configObjDict['num_vel'],
                            nlp_configObjDict['az_width_rad'],nlp_configObjDict['num_az'],
                            nlp_configObjDict['el_width_rad'], nlp_configObjDict['num_el'])
    radar_configDict = json.load(fileJsonradar)
    radar_configObj = radar_config(radar_configDict['RadarConfigData'],
                                radar_configDict['DataAssumptions'],
                                radar_configDict['PreProcessing'])

# load files      
    filename_header = '/'+Process_pipelineObj.input_dir+'/header_scan_001.bin'
    filename_data = '/'+Process_pipelineObj.input_dir+ '/iq_scan_001.bin'

    data_samples_complex,scanHeader = loadFile(filename_header,filename_data)
    CPIindex = 0
    cpi_iq = get_cpi_samples(data_samples_complex,scanHeader.CPIHeaders[CPIindex])

# create forward operator and adjoint and check the accuracy of adjoint operator
    AZ0 = scanHeader.CPIHeaders[CPIindex].AzElec
    EL0 = scanHeader.CPIHeaders[CPIindex].ElElec
    num_pol=2
    op1 = angleDopplerOperator(nlp_configObj,AZ0,EL0)
    x= torch.randn(1,num_pol,nlp_configObj.num_vel,nlp_configObj.num_el*nlp_configObj.num_az,dtype=torch.complex64)
    Ax = op1.forward_mat(scanHeader.CPIHeaders[CPIindex],scanHeader.Sensor,x,nlp_configObj)
    # y = torch.randn(Ax.shape,dtype=torch.complex64)
    # Aty = op1.adjoint_mat(scanHeader.CPIHeaders[CPIindex],scanHeader.Sensor,y,nlp_configObj)
    # e1 = torch.matmul(torch.conj_physical(Ax.flatten()).T,y.flatten())
    # e2 = torch.matmul(torch.conj_physical(x.flatten()).T,Aty.flatten())
    # err = torch.abs(e1-e2)/torch.max(torch.abs(e1),torch.abs(e2))
    # az=torch.tensor(0.01)
    # el=torch.tensor(0.02)
    # vel = torch.tensor(100)
    # amp=torch.tensor([10,100])
    # y1= op1.generate_response(scanHeader.CPIHeaders[CPIindex],scanHeader.Sensor,az,el,vel,amp)


# %%
