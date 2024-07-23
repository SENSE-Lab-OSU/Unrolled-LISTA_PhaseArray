# %% import packages
import torch
import numpy as np
import scipy.constants as scc
import sys


sys.path+=[1, '/research/nfs_ertin_1/nithin_data/mod/blip/python/src/unrolled/util']


class proximal_group_sparse():
    def __init__(self):
        pass
    
    def group_reg(self,x):
        return torch.sum(torch.linalg.vector_norm(x,ord=2,dim=1),dim=[1,2])
    def group_norm_matrix(self,x):
        return torch.sqrt(torch.sum(torch.pow(torch.abs(x),2),dim=2))     
    def prox(self,x,tau):
        n = torch.sqrt(torch.sum(torch.pow( torch.abs(x),2),dim=1,keepdim=True))     
        return x*(torch.maximum(n-tau,torch.zeros_like(n))/(n+(torch.where(n==0,1,0))))
        
        
#%%        
if __name__ == "__main__":
    x= torch.zeros(1,2,2,2,dtype=torch.complex64)
    layer1 = proximal_group_sparse(1.0,3000)
    n,y = layer1.group_reg(x)
    p = layer1.prox(x)
    
# %%
