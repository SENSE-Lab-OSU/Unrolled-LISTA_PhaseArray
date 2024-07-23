#%% load packages
import torch
import numpy as np
import scipy.constants as scc
import sys
import json 
from scipy.io import loadmat
import hdf5storage
from timeit import default_timer as timer

sys.path+=['/research/nfs_ertin_1/nithin_data/mod/blip/python/src/unrolled/util',
           '/research/nfs_ertin_1/nithin_data/mod/blip/python/src/unrolled/ForwardOperator']
from load_bin import nlp_config,radar_config,loadFile,get_cpi_samples,Process_pipelineDecoder,Process_pipeline
from proximalOp import proximal_group_sparse
from angle_doppler_operator import angleDopplerOperator
from LISTA_base import LISTA
from LISTA_fista import LISTA_fista
from LISTA_Ifista import LISTA_Ifista

from scipy.io import savemat
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
propSpeed = scc.c
pi = torch.acos(torch.zeros(1)).item() * 2

#%% load files
fileJson = open('/research/nfs_ertin_1/nithin_data/mod/blip/config/process/process_Static_Tower_09192023.json')
device = torch.device('cuda:3') #currently using GPU-1
torch.cuda.set_device(3) #currently using GPU-1
numTargets = 6
numInstances = 20

fileSave = '/research/nfs_ertin_1/nithin_data/mod/blip/python/src/unrolled/simulaed_data/' + 'angle_doppler_targets_{0}.mat'.format(numTargets)
CPIindex=0

X = loadmat(fileSave)
# %%
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

_,scanHeader = loadFile(filename_header,filename_data)


# create forward operator and adjoint and check the accuracy of adjoint operator
AZ0 = scanHeader.CPIHeaders[CPIindex].AzElec
EL0 = scanHeader.CPIHeaders[CPIindex].ElElec
num_pol=2
op1 = angleDopplerOperator(nlp_configObj,gpuDev=device)

#%% 
scale_mag = 45.6228

#%% 
total_iter = 100
train_size = 200
mini_batch_size=20
test_size = 500
for numLayers in np.arange(90,101,5):
    # Z_1 = np.zeros((1,numInstances,2,nlp_configObj.num_vel,
    #                 nlp_configObj.num_az*nlp_configObj.num_el),dtype=np.complex64)
    Z_2 = np.zeros((1,test_size,2,nlp_configObj.num_vel,
                     nlp_configObj.num_az*nlp_configObj.num_el),dtype=np.complex64)
    Z_3 = np.zeros((1,test_size,2,nlp_configObj.num_vel,
                     nlp_configObj.num_az*nlp_configObj.num_el),dtype=np.complex64)
    Z_2_debias = np.zeros((1,test_size,2,nlp_configObj.num_vel,
                     nlp_configObj.num_az*nlp_configObj.num_el),dtype=np.complex64)
    Z_3_debias = np.zeros((1,test_size,2,nlp_configObj.num_vel,
                     nlp_configObj.num_az*nlp_configObj.num_el),dtype=np.complex64)
    # y_ret_1 = np.zeros((1,numInstances,2,32,24),dtype=np.complex64)
    y_ret_2 = np.zeros((1,test_size,2,32,24),dtype=np.complex64)
    y_ret_3 = np.zeros((1,test_size,2,32,24),dtype=np.complex64)
    y_ret_debias_2 = np.zeros((1,test_size,2,32,24),dtype=np.complex64)
    y_ret_debias_3 = np.zeros((1,test_size,2,32,24),dtype=np.complex64)
    # times_1 = np.zeros((1,numInstances))
    times_2 = np.zeros((1,test_size))
    times_3 = np.zeros((1,test_size))
    # MSE_1 = np.zeros((1,numInstances))
    MSE_2 = np.zeros((1,test_size))
    MSE_3 = np.zeros((1,test_size))
    # G_1 = np.zeros((1,numInstances))
    G_2 = np.zeros((1,test_size))
    G_3 = np.zeros((1,test_size))
    # loss_1_store=np.zeros((1,numInstances,total_iter))
    loss_2_store=np.zeros((1,numInstances,total_iter))
    loss_3_store=np.zeros((1,numInstances,total_iter))

    #%% IFISTA 
    # model1=LISTA_Ifista( numLayers=numLayers, scale_mag=scale_mag,step_size=5e-2,gpu_dev = device, 
    #                 actfunc="shrink",  angleDopplerOp=op1,tiedRegularization=True,fixRegularizer = True)
    # model1.to(device)
    #%% FISTA 
    model2=LISTA_fista( numLayers=numLayers, scale_mag=scale_mag,step_size=5e-2,gpu_dev = device, 
                actfunc="shrink",  angleDopplerOp=op1,tiedRegularization=True,fixRegularizer = True)
    model2.to(device)
    #%% ISTA 
    model3=LISTA( numLayers=numLayers, scale_mag=scale_mag,step_size=1e-1,gpu_dev = device, 
                actfunc="shrink",  angleDopplerOp=op1,tiedRegularization=True,fixRegularizer=True)
    model3.to(device)
    
    # optimizer_1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    # scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_1, [total_iter//3, total_iter//2], gamma=0.5)
    optimizer_2 = torch.optim.Adam(model2.parameters(), lr=1e-5)
    scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_2, [total_iter//3, total_iter//2], gamma=0.5)
    optimizer_3 = torch.optim.Adam(model3.parameters(), lr=1e-5)
    scheduler_3 = torch.optim.lr_scheduler.MultiStepLR(optimizer_3, [total_iter//3, total_iter//2], gamma=0.5)

    for idxsnr in np.arange(12,13):
        y_train = torch.tensor(X['YSave'][0:train_size,:,:,:] 
                        + X['noiseaddSave'][idxsnr,0:train_size,:,:,:],dtype=torch.complex64)
        y_train=y_train
        y_test = torch.tensor(X['YSave'][0:test_size,:,:,:]
                        + X['noiseaddSave'][idxsnr,0:test_size,:,:,:],dtype=torch.complex64)
        y_test=y_test.to(device)

        for iter in range(total_iter):
            print('{} snr index,{} num layers, {} epoch'.format(idxsnr,numLayers,iter))
            for miniBatch in np.arange(0,200,mini_batch_size):
                y = y_train[miniBatch:miniBatch+mini_batch_size,:,:,:].to(device) 
                Y={"y":y,'cpi_header':scanHeader.CPIHeaders[CPIindex],
                    'sensor':scanHeader.Sensor,'nlp_opts':nlp_configObj}
                # model1.opinit(Y)
                model2.opinit(Y)
                model3.opinit(Y)     
                              

                optimizer_2.zero_grad()
                Z_2_temp,y_ret_2_temp,reg_2_temp,Z_2_temp_debias,y_ret_debias_2_temp = model2.forward(Y)
                
                loss=torch.mean(torch.pow(torch.abs(y_ret_2_temp-y),2)) 
                loss.backward()
                loss_2_store[0,:,iter]=loss.detach().cpu().numpy()
                optimizer_2.step()
                scheduler_2.step()
                print('Fista {} iter,loss={}'.format(iter,loss.detach().cpu().numpy()))

                optimizer_3.zero_grad()
                Z_3_temp,y_ret_3_temp,reg_3_temp,Z_3_temp_debias,y_ret_debias_3_temp = model3.forward(Y)
                loss=torch.mean(torch.pow(torch.abs(y_ret_3_temp-y),2))
                loss.backward()
                loss_3_store[0,:,iter]=loss.detach().cpu().numpy()
                optimizer_3.step()
                scheduler_3.step()
                print('Ista {} iter,loss={}'.format(iter,loss.detach().cpu().numpy()))
        del Z_3_temp,y_ret_3_temp,Z_3_temp_debias,reg_3_temp
        del Z_2_temp,y_ret_2_temp,Z_2_temp_debias,reg_2_temp
        del y 
        torch.cuda.empty_cache()     
        with torch.no_grad():
            start_time=timer()
            for miniBatch in np.arange(0,500,mini_batch_size):
                y = y_test[miniBatch:miniBatch+mini_batch_size,:,:,:].to(device)               
                Y={"y":y,'cpi_header':scanHeader.CPIHeaders[CPIindex],
                        'sensor':scanHeader.Sensor,'nlp_opts':nlp_configObj}
                Z_2_temp,y_ret_2_temp,reg_2_temp,Z_2_temp_debias,y_ret_debias_2_temp = model2.forward(Y)    
                Z_2[0,miniBatch:miniBatch+mini_batch_size,:,:,:]=np.squeeze(Z_2_temp.detach().cpu().numpy())
                y_ret_2[0,miniBatch:miniBatch+mini_batch_size,:,:,:] =np.squeeze(y_ret_2_temp.detach().cpu().numpy())
                Z_2_debias[0,miniBatch:miniBatch+mini_batch_size,:,:,:]=np.squeeze(Z_2_temp_debias.detach().cpu().numpy())
                y_ret_debias_2[0,miniBatch:miniBatch+mini_batch_size,:,:,:] =np.squeeze(y_ret_debias_2_temp.detach().cpu().numpy())
                
                end_time = timer()
                times_2[0,miniBatch:miniBatch+mini_batch_size] = end_time-start_time
                MSE_2[0,miniBatch:miniBatch+mini_batch_size] = (0.5*(np.sum(np.abs(
                    y_ret_2[0,miniBatch:miniBatch+mini_batch_size,:,:,:]-np.squeeze(
                        y.detach().cpu().numpy()))**2.0,axis=(1,2,3))))
                G_2[0,miniBatch:miniBatch+mini_batch_size] = reg_2_temp.detach().cpu().numpy()

                start_time=timer()
                Z_3_temp,y_ret_3_temp,reg_3_temp,Z_3_temp_debias,y_ret_debias_3_temp = model3.forward(Y)  
                Z_3[0,miniBatch:miniBatch+mini_batch_size,:,:,:]=np.squeeze(Z_3_temp.detach().cpu().numpy())
                y_ret_3[0,miniBatch:miniBatch+mini_batch_size,:,:,:] =np.squeeze(y_ret_3_temp.detach().cpu().numpy())
                y_ret_debias_3[0,miniBatch:miniBatch+mini_batch_size,:,:,:] =np.squeeze(y_ret_debias_3_temp.detach().cpu().numpy())
                                
                end_time = timer()
                times_3[0,miniBatch:miniBatch+mini_batch_size] = end_time-start_time
                MSE_3[0,miniBatch:miniBatch+mini_batch_size] = (0.5*(np.sum(np.abs(y_ret_3[0,miniBatch:miniBatch+mini_batch_size,:,:,:]
                                -np.squeeze(y.detach().cpu().numpy()))**2.0,axis=(1,2,3))))
                G_3[0,miniBatch:miniBatch+mini_batch_size] = reg_3_temp.detach().cpu().numpy()
                stepW_2 = torch.tensor(model2.stepW).detach().cpu().numpy()
                stepW_3 = torch.tensor(model3.stepW).detach().cpu().numpy()

    f1=  '/research/nfs_ertin_1/nithin_data/mod/blip/python/src/unrolled/simulaed_data/'
    vals_to_save = {'y_ret_2':y_ret_2,'Z_2':Z_2,'Z_2_debias':Z_2_debias,'times_2':times_2,'MSE_2':MSE_2,'G_2':G_2,'stepW_2':stepW_2,'y_ret_debias_2':y_ret_debias_2,
        'y_ret_3':y_ret_3,'Z_3':Z_3,'Z_3_debias':Z_3_debias,'times_3':times_3,'MSE_3':MSE_3,'G_3':G_3,'stepW_3':stepW_3,'y_ret_debias_3':y_ret_debias_3}

    hdf5storage.savemat('/research/nfs_ertin_1/nithin_data/mod/blip/python/src/unrolled/simulaed_data/matFinal_trainedNet_test_batchS{}.mat'.format(numLayers), vals_to_save , format='7.3')
#%% 
# Z=Z+1e-5
# fig1, ax1 = plt.subplots()
# im1=ax1.imshow(10*np.log10(np.abs(np.squeeze(Z[0,0,:,:]))),
#                aspect='auto',cmap='plasma',
#                vmin=np.max(10*np.log10(np.abs(np.squeeze(Z[0,0,:,:]))))-10,
#                vmax=np.max(10*np.log10(np.abs(np.squeeze(Z[0,0,:,:])))),
#                extent =[0,9024,nlp_configObj.velocity_min,nlp_configObj.velocity_max])
# ax1.set_title('H-polarization LISTA unoptimized')
# ax1.set_xlabel('Angle bins',fontsize=14)
# ax1.set_ylabel('Doppler velocity',fontsize=14)

# fig1.colorbar(im1,ax=ax1)

# fig2, ax2 = plt.subplots()
# im2=ax2.imshow(10*np.log10(np.abs(np.squeeze(Z[0,1,:,:]))),
#                aspect='auto',cmap='plasma',vmin=np.max(10*np.log10(np.abs(np.squeeze(Z[0,1,:,:]))))-10,
#                vmax=np.max(10*np.log10(np.abs(np.squeeze(Z[0,1,:,:])))),
#                extent =[0,9024,nlp_configObj.velocity_min,nlp_configObj.velocity_max])
# ax2.set_title('V-polarization LISTA unoptimized')
# ax2.set_xlabel('Angle bins',fontsize=14)
# ax2.set_ylabel('Doppler velocity',fontsize=14)

# fig2.colorbar(im2,ax=ax2)
# fig3, ax3 = plt.subplots()
# Z0 = op1.adjoint_mat(scanHeader.CPIHeaders[CPIindex],scanHeader.Sensor,y,nlp_configObj)
# Z0=Z0.cpu().numpy()
# im3=ax3.imshow(10*np.log10(np.abs(np.squeeze(Z0[0,0,:,:]))),
#                aspect='auto',cmap='plasma',vmin=np.max(10*np.log10(np.abs(np.squeeze(Z0[0,0,:,:]))))-10,
#                vmax=np.max(10*np.log10(np.abs(np.squeeze(Z0[0,0,:,:])))),
#                extent =[0,9024,nlp_configObj.velocity_min,nlp_configObj.velocity_max])
# ax3.set_title('H-polarization Matched filter')
# ax3.set_xlabel('Angle bins',fontsize=14)
# ax3.set_ylabel('Doppler velocity',fontsize=14)

# fig3.colorbar(im3,ax=ax3)
# fig4, ax4 = plt.subplots()
# im4 =ax4.imshow(10*np.log10(np.abs(np.squeeze(Z0[0,1,:,:]))),
#                 aspect='auto',cmap='plasma',vmin=np.max(10*np.log10(np.abs(np.squeeze(Z0[0,1,:,:]))))-10,
#                vmax=np.max(10*np.log10(np.abs(np.squeeze(Z0[0,1,:,:])))),
#                extent =[0,9024,nlp_configObj.velocity_min,nlp_configObj.velocity_max])
# ax4.set_title('V-polarization Matched filter')
# ax4.set_xlabel('Angle bins',fontsize=14)
# ax4.set_ylabel('Doppler velocity',fontsize=14)
# fig4.colorbar(im4,ax=ax4)
# savemat(f1,vals_to_save)
# # %%

# Z_vel= np.max(np.abs(Z),axis=(0,1,3))
# Z0_vel= np.max(np.abs(Z0),axis=(0,1,3))

# vel_grid= np.linspace(nlp_configObj.velocity_min,
#                       nlp_configObj.velocity_max,nlp_configObj.num_vel) 
# fig5, ax5 = plt.subplots()
# plt1 =ax5.plot(vel_grid,Z_vel)
# ax5.set_title('Doppler estimation')
# ax5.set_xlabel('range rate m/s',fontsize=14)
# ax5.set_ylabel('rcs',fontsize=14)

# vv=X['velSave']
# vp=ax5.vlines(vv[idxRealization,0], 0, 120, colors='r', linestyles='dotted')
# ax5.vlines(vv[idxRealization,1], 0, 120, colors='r', linestyles='dotted')
# ax5.vlines(vv[idxRealization,2], 0, 120, colors='r', linestyles='dotted')
# ax5.vlines(vv[idxRealization,3], 0, 120, colors='r', linestyles='dotted')
# ax5.vlines(vv[idxRealization,4], 0, 120, colors='r', linestyles='dotted')
# ax5.vlines(vv[idxRealization,5], 0, 120, colors='r', linestyles='dotted')
# ax5.legend([plt1,vp],['LISTA recovered','Ground Truth'])

# fig6, ax6 = plt.subplots()
# plt2 =ax6.plot(vel_grid,Z0_vel)
# ax6.set_title('Doppler estimation Matched filter')
# ax6.set_xlabel('range rate m/s',fontsize=14)
# ax6.set_ylabel('rcs',fontsize=14)


# vp2=ax6.vlines(vv[0,0], 0,200, colors='r', linestyles='dotted')
# ax6.vlines(vv[0,1], 0, 200, colors='r', linestyles='dotted')
# ax6.vlines(vv[0,2], 0, 200, colors='r', linestyles='dotted')
# ax6.vlines(vv[0,3], 0, 200, colors='r', linestyles='dotted')
# ax6.vlines(vv[0,4], 0, 200, colors='r', linestyles='dotted')
# ax6.vlines(vv[0,5], 0, 200, colors='r', linestyles='dotted')
# ax6.legend([plt2,vp2],['Matched filtered ','Ground Truth'])


# # %%
# vpeaks,_=find_peaks(x=Z_vel,prominence=20000)
# angleDetects = Z[0,:,vpeaks,:]
# distances=cdist(vel_grid[vpeaks].reshape((-1,1)),vv[idxRealization,:].reshape((-1,1)))
# distancesMin = np.min(distances,axis=1)
# idxMin = np.argmin(distances,axis=1)

# distancesThreshold = distancesMin[distancesMin<0.3]
# idxMinThresholdgroundtruth=idxMin[distancesMin<0.3]
# idxPlot = np.where(distancesMin<0.3)[0]
# fig={}
# ax={}
# im={}
# truePlot={}
# imCount=0
# for i in idxPlot:
#    angPlot=np.squeeze(angleDetects[i,1,:]).reshape(nlp_configObj.num_az,nlp_configObj.num_el)+1e-6
#    fig[imCount], ax[imCount] = plt.subplots()
#    im[imCount]=ax[imCount].imshow(10*np.log10(np.abs(angPlot)),
#                aspect='auto',cmap='plasma',vmin=np.max(10*np.log10(np.abs(angPlot)))-6,
#                vmax=np.max(10*np.log10(np.abs(angPlot))),
#                extent =[AZ0+np.min(op1.Azgrid.numpy()),AZ0+np.max(op1.Azgrid.numpy()),
#                         EL0+np.min(op1.Elgrid.numpy()),EL0+np.max(op1.Elgrid.numpy())])
#    ax[imCount].set_title('target detection')
#    ax[imCount].set_xlabel('Azimuth radians',fontsize=14)
#    ax[imCount].set_ylabel('Elevation radians',fontsize=14)
#    truePlot[imCount]=ax[imCount].scatter(X['azSave'][idxRealization,idxMinThresholdgroundtruth[imCount]],
#                  X['elSave'][idxRealization,idxMinThresholdgroundtruth[imCount]],200,'g')
#    ax[imCount].legend([im[imCount],truePlot[imCount]],['LISTA recovered','Ground Truth'])
#    fig[imCount].colorbar(im[imCount],ax=ax[imCount])
#    imCount+=1

# %%
