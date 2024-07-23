# %% import packages
import torch
import numpy as np
import scipy.constants as scc
import sys
import json 
sys.path+=['/research/nfs_ertin_1/nithin_data/mod/blip/python/src/unrolled/util',
           '/research/nfs_ertin_1/nithin_data/mod/blip/python/src/unrolled/ForwardOperator']

from load_bin import nlp_config,radar_config,loadFile,get_cpi_samples,Process_pipelineDecoder,Process_pipeline
from proximalOp import proximal_group_sparse
from angle_doppler_operator import angleDopplerOperator
from scipy.io import savemat
propSpeed = scc.c
pi = torch.acos(torch.zeros(1)).item() * 2

# %%
def generateData_angle_doppler(fileJson,fileSave,numTargets = 5,numInstances = 100,snr=12,CPIindex=0):
    Process_pipelineObj = json.load(fileJson,object_hook=Process_pipelineDecoder)

    fileJsonradar = open(Process_pipelineObj.radar_config)
    fileJsonnlp = open(Process_pipelineObj.nlp_config)

    nlp_configObjDict = json.load(fileJsonnlp)
    nlp_configObj = nlp_config(nlp_configObjDict['range_min'], nlp_configObjDict['range_max'],
                            nlp_configObjDict['range_oversample'],nlp_configObjDict['velocity_min'],
                            nlp_configObjDict['velocity_max'],nlp_configObjDict['num_vel'],
                            nlp_configObjDict['az_width_rad'],nlp_configObjDict['num_az'],
                            nlp_configObjDict['el_width_rad'], nlp_configObjDict['num_el'])


    # load files      
    filename_header = '/'+Process_pipelineObj.input_dir+'/header_scan_001.bin'
    filename_data = '/'+Process_pipelineObj.input_dir+ '/iq_scan_001.bin'

    _,scanHeader = loadFile(filename_header,filename_data)
   

    AZ0 = scanHeader.CPIHeaders[CPIindex].AzElec
    EL0 = scanHeader.CPIHeaders[CPIindex].ElElec
    
    op1 = angleDopplerOperator(nlp_configObj,AZ0,EL0)
    numPulses = len(scanHeader.CPIHeaders[CPIindex].CenterFrequency)
    numPol=2
    numChannels = len(scanHeader.CPIHeaders[CPIindex].ChannelIDs)
    Y =torch.zeros(numInstances,numPol,numPulses,numChannels,dtype=torch.complex64)
    noiseadd=torch.zeros( snr.size+1,numInstances,numPol,numPulses,numChannels,dtype=torch.complex64)
    amps=100*torch.randn(numInstances,numTargets,numPol,dtype=torch.complex64)
    vel= ( nlp_configObj.velocity_max- nlp_configObj.velocity_min) * torch.rand(numInstances, numTargets) + nlp_configObj.velocity_min
    az = nlp_configObj.az_width_rad*torch.rand(numInstances, numTargets)+ AZ0-nlp_configObj.az_width_rad/2
    el = nlp_configObj.el_width_rad*torch.rand(numInstances, numTargets)+ EL0-nlp_configObj.el_width_rad/2
    normY = torch.zeros(numInstances,1)
    normN = torch.zeros(snr.size+1,numInstances,1)
    
    for idxRealization in torch.arange(numInstances):
        for idxTargets in torch.arange(numTargets):
            Y[idxRealization,:,:,:]+=op1.generate_response(scanHeader.CPIHeaders[CPIindex],
                                                        scanHeader.Sensor,az[idxRealization,idxTargets],
                                                       el[idxRealization,idxTargets],vel[idxRealization,
                                                        idxTargets],amps[idxRealization,idxTargets,:]).squeeze()
        normY[idxRealization] = torch.linalg.vector_norm(Y[idxRealization,:,:,:])/(torch.numel(Y[idxRealization,:,:,:])**0.5)
        idxsnr = 0
        for snrVal in snr:
            stdNoise = 10**(-snrVal/10)*normY[idxRealization]
            noiseadd[idxsnr,idxRealization,:,:,:] = torch.normal(mean =0,std=stdNoise[0],size=(Y[idxRealization,:,:,:].shape[0],Y[idxRealization,:,:,:].shape[1],Y[idxRealization,:,:,:].shape[2]))
            normN[idxsnr,idxRealization] = torch.linalg.vector_norm(noiseadd[idxsnr,idxRealization,:,:,:])/(torch.numel(noiseadd[idxsnr,idxRealization,:,:,:])**0.5)
            idxsnr=idxsnr+1
    
    azSave = az.numpy()
    elSave = el.numpy()
    ampsSave = amps.numpy()
    velSave = vel.numpy()
    YSave = Y.numpy()
    noiseaddSave = noiseadd.numpy()
    vals_to_save = {'azSave':azSave,'elSave':elSave,'ampsSave':ampsSave,
                    'velSave':velSave,'YSave':YSave,'noiseaddSave':noiseaddSave}

    savemat(fileSave,vals_to_save)    
    
    
    
    
    return azSave,elSave,ampsSave,velSave,YSave,noiseaddSave

# %%
if __name__ == "__main__":
    fileJson = open('/research/nfs_ertin_1/nithin_data/mod/blip/config/process/process_Static_Tower_09192023.json')
    
    numTargets = 6
    numInstances = 500
    snr=np.linspace(-10,20,num=20)
    fileSave = '/research/nfs_ertin_1/nithin_data/mod/blip/python/src/unrolled/simulaed_data/' + 'angle_doppler_targets_{0}.mat'.format(numTargets)
    CPIindex=0
    azSave,elSave,ampsSave,velSave,YSave,noiseaddSave = generateData_angle_doppler(fileJson,fileSave,numTargets=numTargets,numInstances=numInstances,snr=snr,CPIindex=CPIindex)

# %%
