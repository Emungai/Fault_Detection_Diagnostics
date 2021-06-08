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
from sklearn.model_selection import train_test_split

import plotly.graph_objs as go

#appending a path
sys.path.append('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/utils')

from RPCA_funcs import RPCA
from similarity_mat import sim_matrix
from LoadFeatures import getDigitFeatures
from LoadFeatures import getDigitState
from plotting_funcs import plotPCA
from PCA_funcs import getPCAComponents

import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

#VARIABLES
forceAxis="x"
run_pca=1
dendogram=1
allfeatModel=1
plot_cluster=0
plot_dendogram=0
write_to_text=1
#plotting
plot_pca=0

#loading data
feat_info={}
feat_info['diff_sampleRate']=1
feat_info['all_feat']=1
 


#loading dataa
feat_info['forceAxis']=forceAxis
# feat_info['file_name']='100N_4-10-21.mat'
# feat_info['file_name']='-100N_4-25-21.mat'
# feat_info['file_name']='100N_noise_5-6-21'
feat_info['file_name']='70N_5-6-21'

[X,time_data_train]=getDigitState(feat_info)
time_data_train_max=max(np.diff(time_data_train))
# data=getDigitFeatures(feat_info)
y= np.zeros(X.shape[0])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# scale data
t = MinMaxScaler()
t.fit(X)
X_train = t.transform(X)


feat_info['file_name']='100N_4-10-21.mat'
[X,time_data_test]=getDigitState(feat_info)
time_data_test_max=max(np.diff(time_data_test))
# scale data
t = MinMaxScaler()
t.fit(X)
X_test = t.transform(X)

len_X_test=X_test.shape[0]
len_X_train=X_train.shape[0]

len_X_min=min(len_X_test,len_X_train)
X_test=X_test[0:len_X_min,]
X_train=X_train[0:len_X_min,]
X_val=X_train

time_max=max(time_data_train_max,time_data_test_max)

filename_test='NEwDigit_test_x.csv'
filename_train='Digit_train_x.csv'
filename_val='Digit_val_x.csv'

np.savetxt(filename_test,X_test,fmt='%.14f',delimiter=',')
np.savetxt(filename_train,X_train,fmt='%.14f',delimiter=',')
np.savetxt(filename_val,X_val,fmt='%.14f',delimiter=',')




