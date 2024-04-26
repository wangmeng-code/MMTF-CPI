# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:53:17 2022

@author: toxicant
"""

import os
import pickle
import pandas as pd
import numpy as np
import torch


def normalize(input_x):
    mean = torch.reshape(torch.mean(input_x, dim=1),(-1,1))
    sub_v = input_x - mean
    pred = torch.sum(torch.pow(sub_v,2), dim=1)
    norm = torch.pow(pred,0.5)

    return norm, sub_v

def norm_laplacian(adj, A):
    """ normalized Laplacian """
    D= np.array(A).sum(axis=1)

    D_inv = np.divide(1., np.sqrt(D), out=np.zeros_like(D), where=D!=0.)
    D_inv_diag=np.zeros_like(adj)

    np.fill_diagonal(D_inv_diag,D_inv)
    adj = D_inv_diag.dot(adj).dot(D_inv_diag)

    return adj 

def decomposition():
    ppi_adj = np.array(pd.read_csv('./data/PPI_adj.csv',index_col=0))
    A = ppi_adj
    D= np.array(A).sum(axis=1)
    D1=np.zeros_like(A)
    np.fill_diagonal(D1,D)
    L=D1-A
    L=np.array(norm_laplacian(L, A),dtype=np.float32)
    
    eigenvalue,featurevector=np.linalg.eig(L)
    U=np.real(featurevector)
    U_T=U.T

    return torch.from_numpy(U), torch.from_numpy(U_T)



