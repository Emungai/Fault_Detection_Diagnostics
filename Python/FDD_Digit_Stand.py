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
from sklearn.preprocessing import MinMaxScaler
from itertools import cycle, islice
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as metrics

import plotly.graph_objs as go
from tensorflow.keras.models import load_model

import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

#appending a path
sys.path.append('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/utils')

from RPCA_funcs import RPCA
from similarity_mat import sim_matrix
# import AutoEncoder



#### set variables
diff_sampleRate=1
#clustering variables
min_point=200 #for DBSCAN and OPTICS
affinity_propagation_info=1
DBSCAN_info=0
#plotting variables
plot_pca=1
time_plot=1
escapetime_plot=0
jet_cl=1

forceAxis="y"
standardize_data=0

run_pca=1
run_encoder=0
dendogram=0
#loading data
#see https://docs.scipy.org/doc/scipy/reference/tutorial/io.html for reference
if forceAxis == "y":
    mat=spio.loadmat('/home/exo/Documents/eva/Fault_Detection_Diagnostics/FDD_Analysis/data/digit/Biped_Controller/y_force_act_moment/100N_4-10-21.mat', struct_as_record=False, squeeze_me=True)
elif forceAxis == "x":
    mat=spio.loadmat('/home/exo/Documents/eva/Fault_Detection_Diagnostics/FDD_Analysis/data/digit/Biped_Controller/x_force_act_moment/100N_4-10-21.mat', struct_as_record=False, squeeze_me=True)

#get features
# region
logger=mat['logger']
q_all=logger.q_all
dq_all=logger.dq_all
ua_all=logger.ua_all
ud_all=logger.ud_all
LG=logger.LG
L_LeftFoot=logger.L_LeftFoot
L_RightFoot=logger.L_RightFoot
rp_COMFoot=logger.rp_COMFoot
task=logger.task
time_data=logger.time
feat=logger.feat_names
p_LeftFoot=logger.p_LeftFoot
rpy_LeftFoot=logger.rpy_LeftFoot
p_RightFoot=logger.p_RightFoot
rpy_RightFoot=logger.rpy_RightFoot
p_com=logger.p_com
p_links=logger.p_links

if forceAxis == "y":
    q_idx=np.array([2,3,6,7,11,13,14,18,22,24,25])-1
    u_idx=np.array([1,5,6,7,11,12,13,17])-1
    L_idx=0
    p_idx=np.array([2,3])-1

elif forceAxis== "x":
    q_idx=np.array([1,3,5,9,10,11,12,15,17,20,21,22,23,26,28])-1
    u_idx=np.array([3,4,5,6,9,10,11,10,14,15,18,20])-1
    L_idx=1
    p_idx=np.array([1,3])-1  

q_all_f=np.array(q_all[q_idx,])
dq_all_f=np.array(dq_all[q_idx,])
ua_all_f=np.array(ua_all[u_idx,])
ud_all_f=np.array(ud_all[u_idx,])
LG_all_f=np.array(LG[L_idx,])
L_LeftFoot_f=np.array(L_LeftFoot[L_idx,])
L_RightFoot_f=np.array(L_RightFoot[L_idx,])
rp_COMFoot_f=np.array(rp_COMFoot[p_idx,])
p_LeftFoot_f=np.array(p_LeftFoot[p_idx,])
rpy_LeftFoot_f=np.rad2deg(np.array(rpy_LeftFoot[L_idx,]))
p_RightFoot_f=np.array(p_RightFoot[p_idx,])
rpy_RightFoot_f=np.rad2deg(np.array(rpy_RightFoot[L_idx,]))
p_com=np.array(p_com)
p_links=np.array(p_links)

# FDD = np.row_stack((q_all_f, dq_all_f,ua_all_f-ud_all_f,LG_all_f,L_LeftFoot_f,L_RightFoot_f,rp_COMFoot_f))
FDD = np.row_stack((q_all_f, dq_all_f,LG_all_f,L_LeftFoot_f,L_RightFoot_f,rp_COMFoot_f))
FDD=np.transpose(FDD)
feet_info=np.transpose(np.row_stack((p_LeftFoot_f,rpy_LeftFoot_f,p_RightFoot_f,rpy_RightFoot_f)))
p_com=np.transpose(p_com)
p_links=np.transpose(p_links)
#endregion


#*****cutting off the part where AR controller's was running***
# region
t=np.array(time_data)-3
idx_bc=np.where(t<0)
idx_end=len(t)
FDD_bc=FDD[idx_bc[0][-1]:idx_end,]
time_data=time_data[idx_bc[0][-1]:idx_end]
feet_info=feet_info[idx_bc[0][-1]:idx_end]
p_links=p_links[idx_bc[0][-1]:idx_end]
p_com=p_com[idx_bc[0][-1]:idx_end]

# endregion


# get different sample rate
if diff_sampleRate:

    FDD_bc=FDD_bc[0::12,:] #to grab everyother column Y[:,0::2]
    time_data=time_data[0::12]
    feet_info=feet_info[0::12]
    p_links=p_links[0::12]
    p_com=p_com[0::12]


#figure out when robot has started to fall and fallen, and create label for classification
#region
label=np.zeros((len(FDD_bc),1))
#robot starting to fall
feet_info_r=np.column_stack((feet_info[:,2],feet_info[:,5]))
feet_info_rb=30- np.abs(feet_info_r)
#first figuring out where feet_info_rb is negaitve since that's where the feet angles are greater than zero
#we do this using feet_info_rb<0 which returns a Boolean (note it'll return true when the feet angles are greater than 30) 
#we then sum the rows of this Boolean, if the feet angles are greater than
#zero, the summation will be greater than 0
r_falling=np.where(np.sum(feet_info_rb<0,axis=1)>0)
idx_falling=r_falling[0][0]
label[idx_falling:len(label):,]=1
#robot has fallen
p_linksSansToes=np.delete(p_links,[6,7,17,18],1)
p_feet=np.concatenate(([feet_info[0,1]],[feet_info[0,4]],p_links[0,[6,7,17,18]])) #initial z positions for: left foot, right foot,left toe pitch link, left toe roll link, right toe pitch link, right toe roll link
p_feet_zmax=np.amax(p_feet)
p_linksSansToesb=p_linksSansToes-p_feet_zmax-0.1 #trying to find where the other limbs are atleast 0.1m away from floor
r_fallen=np.where(np.sum(p_linksSansToesb<0,axis=1)>0)
idx_fallen=r_fallen[0][0]
label[idx_fallen:len(label):,]=2
#endregion

if standardize_data:
    # standardize dataset for easier parameter selection
    X = StandardScaler().fit_transform(FDD_bc) #to see whether there's a nan element np.any(np.isnan(mat))
else:
    # scale dataset
    X=MinMaxScaler().fit_transform(FDD_bc)



if run_pca:
    #RPCA: algorithm is from Brunton's book
    L,S = RPCA(X)
    #PCA see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    pca=PCA(n_components=3) #setting up the PCA function and keeping all the principal components
    pca.fit(L) #applying PCA to L
    V=pca.components_.T #grabbing the principal components
    lam=pca.explained_variance_ratio_ #percentage of variance explained by each of the principal components 

    #transforming the data using the first 3 PCA components. Note can't use Y=np.array(pca.transform(X)) unless we add back in the mean of each column of L  that was subtracted using
    Y = np.matmul( X- np.mean(X,axis=0), V)
    n_clust=3
    eps_db=0.7
elif run_encoder:
    n_clust=4
    eps_db=0.7
    if forceAxis=='y':
        encoder=load_model('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/encoder_minmax_y_100.h5',compile=False)
    elif forceAxis=='x':
        encoder=load_model('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/encoder_minmax_x_100.h5',compile=False) 
    Y=encoder.predict(X)

if dendogram:
    plt.figure(figsize=(10, 7))
    linked=shc.linkage(Y, method='ward')
    c, coph_dists = cophenet(linked, pdist(Y))
    print(c)
    dend = shc.dendrogram(linked)
    plt.show()

#plotting results from PCA
if plot_pca:
    fig = plt.figure()
    fig1=plt.figure()
    fig2=plt.figure()
    fig3=plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax1 = fig1.add_subplot()
    ax2 = fig2.add_subplot()
    ax3 = fig3.add_subplot()

    if time_plot:
        if jet_cl:  
            jet=cm.get_cmap('jet', int(math.modf(time_data[-1])[1])+1)
            cmap=cm.jet
        else:
            viridis=cm.get_cmap('viridis', int(math.modf(time_data[-1])[1])+1)
            viridis_color=viridis.colors
            cmap=cm.viridis
    elif escapetime_plot:
        if jet_cl:  
            jet=cm.get_cmap('jet', int(label[-1])+1)
            cmap=cm.jet
        else:
            viridis=cm.get_cmap('viridis',int(label[-1])+1)
            viridis_color=viridis.colors
            cmap=cm.viridis

    for i in range(0,len(Y)):
        if time_plot:
            idx_t=int(math.modf(time_data[i])[1])
        elif escapetime_plot:
            idx_t=int(label[i])
        x=Y[i,0]
        y=Y[i,1]
        z=Y[i,2]
        if jet:
            ax.scatter(x, y,z,color=jet(idx_t))
            ax1.scatter(x, y,color=jet(idx_t))
            ax2.scatter(x, z,color=jet(idx_t))
            ax3.scatter(y, z,color=jet(idx_t))
            # ax.plot(y, z, 'g+', zdir='x')
            # ax.plot(x, y, 'k+', zdir='z')
            #PCM=ax.scatter(Y[i,1], Y[i,2],color=jet(idx_t),vmin=0, vmax=11)

        
        else:
            ax.scatter(Y[i,0], Y[i,1],Y[i,2],color=viridis_color[idx_t])
    
        
        
    ax.set_xlabel('V1')
    ax.set_ylabel('V2')
    ax.set_zlabel('V3')
    if time_plot:
        norm = cm.colors.Normalize(vmin=0, vmax=int(math.modf(time_data[-1])[1]))
        tick_locator = ticker.MaxNLocator(nbins=int(math.modf(time_data[-1])[1])+1)
    elif escapetime_plot:
        norm = cm.colors.Normalize(vmin=0, vmax=int(label[-1]))
        tick_locator = ticker.MaxNLocator(nbins=int(label[-1])+1)        

    #PCM=ax.get_children()[2]
    cb=fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    
    cb.locator = tick_locator
    cb.update_ticks()

    ax1.set_xlabel('V1')
    ax1.set_ylabel('V2')
 

    ax2.set_xlabel('V1')
    ax2.set_ylabel('V3')

    ax3.set_xlabel('V2')
    ax3.set_ylabel('V3')

    plt.show()



if DBSCAN_info:
    #figuring out the epsilon to use for DBSCAN 
    #References: https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd
    #            https://towardsdatascience.com/explaining-dbscan-clustering-18eaf5c83b31
    # var=np.cumsum(np.round(lam, 3)*100)
    # plt.figure(figsize=(12,6))
    # plt.ylabel('% Variance Explained')
    # plt.xlabel('# of Features')
    # plt.title('PCA Analysis')
    # plt.ylim(0,100.5)
    # plt.plot(var)
    # # plt.show()

    plt.figure(figsize=(10,5))
    nn = NearestNeighbors(n_neighbors=min_point).fit(Y)
    distances, idx = nn.kneighbors(Y)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    # plt.show()

  
    
    db = cluster.DBSCAN(eps=eps_db, min_samples=min_point,metric='manhattan').fit(Y)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(Y, labels))

    optics = cluster.OPTICS(min_samples=20,
                        xi=0.15,
                        min_cluster_size=0.05).fit(Y)

    labels = optics.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(Y, labels))

    pca_df=Y

    Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
    labels = db.labels_
    trace = go.Scatter3d(x=pca_df[:,0], y=pca_df[:,1], z=pca_df[:,2], mode='markers',marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
    layout = go.Layout(scene = Scene, height = 1000,width = 1000)
    data = [trace]
    fig = go.Figure(data = data, layout = layout)
    fig.update_layout(title='DBSCAN clusters (53) Derived from PCA', font=dict(size=12,))
    # fig.show()
else:
    eps_db=0.7 #if min_points=20 use eps_db=0.3



if affinity_propagation_info:
    S=sim_matrix(Y)
    np.fill_diagonal(S,S.min())
    pref=S.min()-200
    # pref=np.median(np.sort(S,axis=None))+100
    # pref=-1000
    # affinity_propagation = cluster.AffinityPropagation(
    #     damping=0.75, preference=int(pref), max_iter=1000,verbose=True).fit(Y)
    # labels = db.labels_
    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)
    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)
    # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(Y, labels))
else:
    pref=-340





fig=plt.figure(figsize=(9 * 2 + 3, 16.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

default_base = {
                # 'quantile': .3,
                'eps': eps_db,
                'damping': .85,
                'preference': int(pref),
                'n_neighbors': 2,
                'n_clusters': n_clust,
                'min_samples_dbscan': min_point,
                'min_samples_optics': 20,
                'xi': 0.15,
                'min_cluster_size': 0.05}

# default_base = {
#                 # 'quantile': .3,
#                 'eps': eps_db,
#                 'damping': .85,
#                 'preference': int(pref),
#                 'n_neighbors': 2,
#                 'n_clusters': 3,
#                 'min_samples': 20,
#                 'xi': 0.25,
#                 'min_cluster_size': 0.1}
params = default_base.copy()

# # estimate bandwidth for mean shift
# bandwidth = cluster.estimate_bandwidth(Y, quantile=params['quantile'])

# connectivity matrix for structured Ward
connectivity = kneighbors_graph(
Y, n_neighbors=params['n_neighbors'], include_self=False)

# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)

# ward = cluster.AgglomerativeClustering(
#         n_clusters=params['n_clusters'], linkage='ward',
#         connectivity=connectivity)

ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward')

# # if forceAxis == "x":        
# spectral = cluster.SpectralClustering(
#     n_clusters=params['n_clusters'], eigen_solver='arpack',
#     affinity="nearest_neighbors",n_neighbors=100)
# elif forceAxis == "y":
spectral = cluster.SpectralClustering(
    n_clusters=params['n_clusters'], eigen_solver='arpack',
    affinity="rbf")

optics = cluster.OPTICS(min_samples=params['min_samples_optics'],
                        xi=params['xi'],
                        min_cluster_size=params['min_cluster_size'])



dbscan = cluster.DBSCAN(eps=params['eps'],min_samples=params['min_samples_dbscan'])

birch = cluster.Birch(n_clusters=params['n_clusters'])

affinity_propagation = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'],max_iter=1000,verbose=True)

# affinity_propagation = cluster.AffinityPropagation(
#         damping=params['damping'],max_iter=1000,verbose=True)


clustering_algorithms = (
    ('AffinityPropagation', affinity_propagation),
    ('SpectralClustering', spectral),
    ('Ward', ward),
    ('DBSCAN', dbscan),
    ('OPTICS', optics),
    ('Birch', birch)
)
# clustering_algorithms = (
#      ('AffinityPropagation', affinity_propagation),
# )

for name, algorithm in clustering_algorithms:
    t0 = time.time()

    # catch warnings related to kneighbors_graph
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="the number of connected components of the " +
            "connectivity matrix is [0-9]{1,2}" +
            " > 1. Completing it to avoid stopping the tree early.",
            category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            message="Graph is not fully connected, spectral embedding" +
            " may not work as expected.",
            category=UserWarning)
        algorithm.fit(Y)

    t1 = time.time()
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(int)
    else:
        y_pred = algorithm.predict(Y)

    labels = algorithm.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Clustering Algorithm: %s' %name)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    if n_clusters_ >1:
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(Y, labels))

    # plt.subplot(1, len(clustering_algorithms), plot_num)
    ax = fig.add_subplot(4, len(clustering_algorithms), plot_num, projection='3d')
    # plot_num += 1
    plot_num1=plot_num+len(clustering_algorithms)
    ax1 = fig.add_subplot(4, len(clustering_algorithms), plot_num1)
    # plot_num += 1
    plot_num2=plot_num1+len(clustering_algorithms)
    ax2 = fig.add_subplot(4, len(clustering_algorithms),plot_num2)
    # plot_num += 1
    plot_num3=plot_num2+len(clustering_algorithms)
    ax3 = fig.add_subplot(4, len(clustering_algorithms),plot_num3)
    # if i_dataset == 0:
    #     plt.title(name, size=18)

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                            '#f781bf', '#a65628', '#984ea3',
                                            '#999999', '#e41a1c', '#dede00']),
                                    int(max(y_pred) + 1))))
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    x=Y[:, 0]
    y=Y[:, 1]
    z=Y[:, 2]
    ax.scatter(x, y,z, s=10, color=colors[y_pred])
    ax.set_xlabel('V1')
    ax.set_ylabel('V2')
    ax.set_zlabel('V3')
    if n_clusters_ >0:
        ax.set_title(name+"\n Clusters:"+ str(n_clusters_)+ " Noise Pts: "+ str(n_noise_)+ "\n Silhoutte:" + str(metrics.silhouette_score(Y, labels)),wrap=True)
    else:
        ax.set_title(name+"\n Clusters:"+ str(n_clusters_)+ " Noise Pts: "+ str(n_noise_),wrap=True)
    ax1.scatter(x, y, s=10, color=colors[y_pred])
    ax1.set_xlabel('V1')
    ax1.set_ylabel('V2')
    # ax1.set_title(name)

    ax2.scatter(x, z, s=10, color=colors[y_pred])
    ax2.set_xlabel('V1')
    ax2.set_ylabel('V3')
    # ax2.set_title(name)

    ax3.scatter(y, z, s=10, color=colors[y_pred])
    ax3.set_xlabel('V2')
    ax3.set_ylabel('V3')
    # ax3.set_title(name)





    # plt.xlim(-2.5, 2.5)
    # plt.ylim(-2.5, 2.5)
    # plt.xticks(())
    # plt.yticks(())
    # plt.zticks(())
    # plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
    #             transform=plt.gca().transAxes, size=15,
    #             horizontalalignment='right')
    plot_num += 1

plt.show()

n_samples=len(Y)
2+2
