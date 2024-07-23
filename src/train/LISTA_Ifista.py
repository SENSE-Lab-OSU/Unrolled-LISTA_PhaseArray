# %% import packages
import torch
import torch.nn as nn
import numpy as np
import sys
import json 
sys.path+=['/research/nfs_ertin_1/nithin_data/mod/blip/python/src/unrolled/util',
           '/research/nfs_ertin_1/nithin_data/mod/blip/python/src/unrolled/ForwardOperator']

from load_bin import nlp_config,radar_config,loadFile,get_cpi_samples,Process_pipelineDecoder,Process_pipeline
from proximalOp import proximal_group_sparse
from angle_doppler_operator import angleDopplerOperator
# %% main lista class model     
class LISTA_Ifista(nn.Module):
    def __init__(self, numLayers, scale_mag, step_size,gpu_dev,actfunc="shrink",  
                 angleDopplerOp=None,tiedRegularization=False,predfinedStepSize=False,
                 fixRegularizer=False):
        super().__init__()
        self.gpu_dev = gpu_dev
        self.angleDopplerOp = angleDopplerOp
        self.NL = numLayers
        self.actfunc = actfunc
        self.scale_mag = scale_mag
        self.step_size = step_size
        self.tiedRegularization = tiedRegularization
        
        if (predfinedStepSize == False)& (fixRegularizer== False):
            stepW,lam = self.initVars()
            self.register_parameter("lam0", lam)
            self.register_parameter("stepW0", stepW)
        elif (predfinedStepSize == True)& (fixRegularizer== False):
            lam=scale_mag
            stepW = step_size
        elif (fixRegularizer == True) & (predfinedStepSize == False):
            lam=scale_mag
            stepW,_ = self.initVars()
            self.register_parameter("stepW0", stepW)
        
        self.lam = [lam]
        self.stepW = [stepW]
        self.proxOp = proximal_group_sparse()
        for L in range(self.NL-1):
            if self.tiedRegularization == False:
                stepW,lam = self.initVars()
                self.register_parameter("stepW%d" % (L + 1), stepW)
                self.stepW.append(stepW)
                
                self.register_parameter("lam%d" % (L + 1), lam)
                self.lam.append(lam)
            else:
                stepW,_ = self.initVars()
                self.register_parameter("stepW%d" % (L + 1),stepW)
                self.stepW.append(stepW)                

    def initVars(self):
        lam_np = self.scale_mag * np.random.rand(1)
        lam = nn.Parameter(torch.tensor(lam_np, dtype=torch.float32, requires_grad=True))
        step_np = self.step_size * np.random.rand(1)
        step_size = nn.Parameter(torch.tensor(step_np, dtype=torch.float32, requires_grad=True))
        
        return step_size,lam 
        
    def applyActFunc(self, x,tau):
        if self.actfunc == "shrink":
            return self.proxOp.prox(x,tau)
            # return torch.sign(x) * torch.clamp(torch.abs(x) - self.lam[L], min=0)
           
            # return torch.sign(x) * torch.clamp(torch.abs(x) - self.lam, min=0)
        # elif self.actfunc == "swish":
        #     return torch.nn.SiLU()(x)
        # elif self.actfunc == "tanh":
        #     return torch.nn.Tanh()(x)
        else:
            raise ValueError('Incorrect activation function set')
    def opinit(self,Y):
        cpi_header = Y['cpi_header']
        sensor = Y['sensor']
        nlp_opts = Y['nlp_opts']
        self.angleDopplerOp.opInit(sensor,nlp_opts,cpi_header)
    def polyProject(self,x,cpi_header,sensor,nlp_opts,mu):
        x_1 = self.angleDopplerOp.gram_matrix(cpi_header,
                                    sensor,x,nlp_opts)*mu
        x_2 = self.angleDopplerOp.gram_matrix(cpi_header,
                                    sensor,x_1,nlp_opts)*(mu**2.0)
        x_3 = self.angleDopplerOp.gram_matrix(cpi_header,
                                    sensor,x_2,nlp_opts)*(mu**3.0)
        x_4 = self.angleDopplerOp.gram_matrix(cpi_header,
                                    sensor,x_3,nlp_opts)*(mu**4.0)
        return 5*x - 10*x_1+10*x_2-5*x_3+x_4
        
    def forward(self, Y):
        y= Y['y']
        cpi_header = Y['cpi_header']
        sensor = Y['sensor']
        nlp_opts = Y['nlp_opts']
        y.detach()
        alphaStep0=torch.tensor(1.0)
        Y1 = torch.zeros_like(self.angleDopplerOp.adjoint_mat(cpi_header,sensor,y,nlp_opts))
        Z0 = torch.zeros_like(self.angleDopplerOp.adjoint_mat(cpi_header,sensor,y,nlp_opts))
        Z1 = self.applyActFunc(Y1-torch.pow(self.stepW[0],2)*self.polyProject(self.angleDopplerOp.least_squares_gram(cpi_header,
                                    sensor,Y1,y.detach(),nlp_opts),cpi_header,sensor,nlp_opts,2.0/160.0),torch.pow(self.lam[0]*self.stepW[0],2))
        alphaStep1=(1+torch.sqrt(1+4*torch.pow(alphaStep0,2.0)))/2.0
        Y1 = Z1+ (alphaStep0-1)/(alphaStep1) * (Z1-Z0)
         
        for L in range(self.NL-1):
            if self.tiedRegularization == False:
                alphaStep0=torch.empty_like(alphaStep1).copy_(alphaStep1)
                Z0 = torch.empty_like(Z1).copy_(Z1)
                Z1 = self.applyActFunc(Y1-torch.pow(self.self.stepW[L+1],2)*self.polyProject(self.angleDopplerOp.least_squares_gram(cpi_header,
                        sensor,Y1,y.detach(),nlp_opts),cpi_header,sensor,nlp_opts,2.0/160.0),torch.pow(self.lam[L+1]*self.stepW[L+1],2))
                alphaStep1=(1+torch.sqrt(1+4*torch.pow(alphaStep0,2)))/2.0
                Y1 = Z1+ (alphaStep0-1)/(alphaStep1) * (Z1-Z0)
            else:
                alphaStep0=torch.empty_like(alphaStep1).copy_(alphaStep1)
                Z0 = torch.empty_like(Z1).copy_(Z1)
                Z1 = self.applyActFunc(Y1-torch.pow(self.stepW[L+1],2)*self.polyProject(self.angleDopplerOp.least_squares_gram(cpi_header,
                    sensor,Y1,y.detach(),nlp_opts),cpi_header,sensor,nlp_opts,2.0/160.0),torch.pow(self.lam[0]*self.stepW[L+1],2))
                alphaStep1=(1+torch.sqrt(1+4*torch.pow(alphaStep0,2)))/2.0
                Y1 = Z1+ (alphaStep0-1)/(alphaStep1) * (Z1-Z0)             
            #torch.cuda.empty_cache()
            
            #print(L)
            #dump_tensors()
        sol = self.angleDopplerOp.forward_mat(cpi_header,sensor,Y1,nlp_opts)
        reg = torch.pow(self.lam[0],2)*self.proxOp.group_reg(Y1)
        return Y1,sol,reg

def pretty_size(size):
        assert isinstance(size, torch.Size)
        return " x ".join(map(str, size))


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc

    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print(
                        "%s:%s%s %s"
                        % (
                            type(obj).__name__,
                            " GPU" if obj.is_cuda else "",
                            " pinned" if obj.is_pinned else "",
                            pretty_size(obj.size()),
                        )
                    )
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print(
                        "%s → %s:%s%s%s%s %s"
                        % (
                            type(obj).__name__,
                            type(obj.data).__name__,
                            " GPU" if obj.is_cuda else "",
                            " pinned" if obj.data.is_pinned else "",
                            " grad" if obj.requires_grad else "",
                            " volatile" if obj.volatile else "",
                            pretty_size(obj.data.size()),
                        )
                    )
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", total_size)
