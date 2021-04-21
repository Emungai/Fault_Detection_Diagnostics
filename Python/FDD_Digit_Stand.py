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



#### set variables
jet_cl=1
diff_sampleRate=1
plot_pca=0
min_point=200
affinity_propagation_info=1
DBSCAN_info=0
plot_pca=0

forceAxis="x"
#loading data
#see https://docs.scipy.org/doc/scipy/reference/tutorial/io.html for reference
if forceAxis == "y":
    mat=spio.loadmat('/home/exo/Documents/eva/Fault_Detection_Diagnostics/FDD_Analysis/data/digit/Biped_Controller/y_force_act_moment/100N_4-10-21.mat', struct_as_record=False, squeeze_me=True)
elif forceAxis == "x":
    mat=spio.loadmat('/home/exo/Documents/eva/Fault_Detection_Diagnostics/FDD_Analysis/data/digit/Biped_Controller/x_force_act_moment/100N_4-10-21.mat', struct_as_record=False, squeeze_me=True)

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
rpy_LeftFoot=logger.rpy_LeftFoot
rpy_RightFoot=logger.rpy_RightFoot

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
rpy_LeftFoot_f=np.rad2deg(np.array(rpy_LeftFoot[L_idx,]))
rpy_RightFoot_f=np.rad2deg(np.array(rpy_RightFoot[L_idx,]))

# FDD = np.row_stack((q_all_f, dq_all_f,ua_all_f-ud_all_f,LG_all_f,L_LeftFoot_f,L_RightFoot_f,rp_COMFoot_f))
FDD = np.row_stack((q_all_f, dq_all_f,LG_all_f,L_LeftFoot_f,L_RightFoot_f,rp_COMFoot_f))
FDD=np.transpose(FDD)

#cutting off the part where AR controller's was running
t=np.array(time_data)-3
idx_bc=np.where(t<0)
idx_end=len(t)
FDD_bc=FDD[idx_bc[0][-1]:idx_end,]
time_data=time_data[idx_bc[0][-1]:idx_end]
rpy_LeftFoot_f=rpy_LeftFoot_f[idx_bc[0][-1]:idx_end]
rpy_RightFoot_f=rpy_RightFoot_f[idx_bc[0][-1]:idx_end]

# #cutting off the part where robot falls 
# t=np.array(time_data)-10.1
# idx_bc=np.where(t<0)
# idx_end=len(t)
# FDD_bc=FDD[0:idx_bc[0][-1]]
# time_data=time_data[0:idx_bc[0][-1]]

if diff_sampleRate:

    FDD_bc=FDD_bc[0::12,:] #to grab everyother column Y[:,0::2]
    time_data=time_data[0::12]
    rpy_LeftFoot_f=rpy_LeftFoot_f[0::12]
    rpy_RightFoot_f=rpy_RightFoot_f[0::12]

 # normalize dataset for easier parameter selection
FDD_bc_s = StandardScaler().fit_transform(FDD_bc) #to see whether there's a nan element np.any(np.isnan(mat))

    #different sampling time


# #RPCA: algorithm is from Brunton's book
L,S = RPCA(FDD_bc_s)



# X=FDD_bc
# L=X


#PCA see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
pca=PCA(n_components=3) #setting up the PCA function and keeping all the principal components
pca.fit(L) #applying PCA to L
V=pca.components_.T #grabbing the principal components
lam=pca.explained_variance_ratio_ #percentage of variance explained by each of the principal components 

Y=np.array(pca.transform(FDD_bc_s)) #transforming the data using the first 3 PCA components

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

    if jet_cl:  
        jet=cm.get_cmap('jet', int(math.modf(time_data[-1])[1])+1)
        cmap=cm.jet
    else:
        viridis=cm.get_cmap('viridis', int(math.modf(time_data[-1])[1])+1)
        viridis_color=viridis.colors
        cmap=cm.viridis

    for i in range(0,len(Y)):
        idx_t=int(math.modf(time_data[i])[1])
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
    norm = cm.colors.Normalize(vmin=0, vmax=int(math.modf(time_data[-1])[1]))

    #PCM=ax.get_children()[2]
    cb=fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    tick_locator = ticker.MaxNLocator(nbins=int(math.modf(time_data[-1])[1])+1)
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
    var=np.cumsum(np.round(lam, 3)*100)
    plt.figure(figsize=(12,6))
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Analysis')
    plt.ylim(0,100.5)
    plt.plot(var)
    # plt.show()

    plt.figure(figsize=(10,5))
    nn = NearestNeighbors(n_neighbors=min_point).fit(Y)
    distances, idx = nn.kneighbors(Y)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    # plt.show()

  
    eps_db=0.7
    db = cluster.DBSCAN(eps=eps_db, min_samples=min_point).fit(Y)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(Y, labels))

    optics = cluster.OPTICS(min_samples=min_point,
                        xi=0.05,
                        min_cluster_size=0.3).fit(Y)
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

## dbscan
X = Y

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
                'n_clusters': 3,
                'min_samples': 20,
                'xi': 0.25,
                'min_cluster_size': 0.1}

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
# bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

# connectivity matrix for structured Ward
connectivity = kneighbors_graph(
X, n_neighbors=params['n_neighbors'], include_self=False)

# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)

ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
if forceAxis == "x":        
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
elif forceAxis == "y":
    spectral = cluster.SpectralClustering(
        n_clusters=3, eigen_solver='arpack',
        affinity="rbf")

# optics = cluster.OPTICS(min_samples=params['min_samples'],
#                         xi=params['xi'],
#                         min_cluster_size=params['min_cluster_size'])
optics = cluster.OPTICS(min_samples=params['min_samples'],
                        xi=params['xi'])


dbscan = cluster.DBSCAN(eps=params['eps'],min_samples=params['min_samples'])

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
        algorithm.fit(X)

    t1 = time.time()
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(int)
    else:
        y_pred = algorithm.predict(X)

    labels = algorithm.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Clustering Algorithm: %s' %name)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
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
    x=X[:, 0]
    y=X[:, 1]
    z=X[:, 2]
    ax.scatter(x, y,z, s=10, color=colors[y_pred])
    ax.set_xlabel('V1')
    ax.set_ylabel('V2')
    ax.set_zlabel('V3')
    ax.set_title(name+"\n Clusters:"+ str(n_clusters_)+ " Noise Pts: "+ str(n_noise_)+ "\n Silhoutte:" + str(metrics.silhouette_score(Y, labels)),wrap=True)
    # ax.set_title(name+"\n Clusters:"+ str(n_clusters_)+ " Noise Pts: "+ str(n_noise_),wrap=True)
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

n_samples=len(X)
2+2
