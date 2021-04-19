import numpy as np
import os
import scipy.io

def sim_matrix(x):
    x=np.array(x)
    S = np.zeros((x.shape[0], x.shape[0]))
    for j in range(x.shape[0]):
        for i in range(j):
            a=x[i,:]
            b=x[j,:]
            S[i,j]= -(np.linalg.norm(a-b))**2
            S[j,i]=S[i,j]
    return S
