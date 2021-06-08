import scipy.io as spio
import numpy as np
import sys
import math


import matplotlib.pyplot as plt
#importing things for color map see https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.pyplot import clim
from matplotlib import ticker

import time
import warnings

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as metrics

import plotly.graph_objs as go

#appending a path
sys.path.append('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/utils')

from RPCA_funcs import RPCA
from similarity_mat import sim_matrix
from LoadFeatures import getDigitFeatures
from plotting_funcs import plotPCA

import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler



def getPCAComponents(pca_info):

    #VARIABLES

    load_singleFile=pca_info['load_singleFile']
    feat_info={}
    feat_info['diff_sampleRate']=pca_info['diff_sampleRate']
    feat_info['all_feat']=pca_info['all_feat']
    standardize_data=pca_info['standardize_data']

    #loading dataa
    if load_singleFile:
        forceAxis=pca_info['forceAxis']
        feat_info['forceAxis']=forceAxis
        feat_info['file_name']=pca_info['file_name']
        data=getDigitFeatures(feat_info)
        time_data = data['time_data']
        FDD_bc = data['FDD_bc']
        label= data['label']
    else:
        file_names=['100N_4-10-21.mat','-100N_4-25-21.mat']
        forceAxis=['x','y']
        FDD_bc_list=[]
        time_data_list=[]
        label_list=[]

        for axis in forceAxis:
            for name in file_names:
                feat_info['forceAxis']=axis
                feat_info['file_name']=name
                data=getDigitFeatures(feat_info) 
                FDD_bc_list.append(data['FDD_bc'])
                time_data_list.append(data['time_data'])
                label_list.append(data['label'])
        FDD_bc=np.concatenate(FDD_bc_list,axis=0)
        time_data=np.concatenate(time_data_list,axis=0)
        label=np.concatenate(label_list,axis=0)

    if standardize_data:
        # standardize dataset for easier parameter selection
        X = StandardScaler().fit_transform(FDD_bc) #to see whether there's a nan element np.any(np.isnan(mat))
    else:
        # scale dataset
        X=MinMaxScaler().fit_transform(FDD_bc)

    #RPCA: algorithm is from Brunton's book
    L,S = RPCA(X)
    #PCA see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    pca=PCA(n_components=3) #setting up the PCA function and keeping all the principal components
    pca.fit(L) #applying PCA to L
    V=pca.components_.T #grabbing the principal components
    lam=pca.explained_variance_ratio_ #percentage of variance explained by each of the principal components 

    # #transforming the data using the first 3 PCA components. Note can't use Y=np.array(pca.transform(X)) unless we add back in the mean of each column of L  that was subtracted using
    # Y = np.matmul( X- np.mean(X,axis=0), V)

    return V
