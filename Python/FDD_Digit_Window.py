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
write_to_text=0
#plotting
plot_pca=1

#getting offline model for PCA
if run_pca:
    pca_info={}
    pca_info['load_singleFile']=0
    pca_info['diff_sampleRate']=1
    pca_info['all_feat']=1
    pca_info['standardize_data']=0
    V=getPCAComponents(pca_info)


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

data=getDigitFeatures(feat_info)


time_data = data['time_data']
FDD_bc = data['FDD_bc']
label= data['label']

#graphing plot wrt
plot_info={}
plot_info['time_plot']=0
plot_info['escapetime_plot']=1
#color pallete of plot
plot_info['jet_cl']=1 #if true, will plot using jet color map otherwise, we'll use viradis
# if run_pca:
#     plot_info['save_file']='/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/figures/moving_window'
# else:
plot_info['save_file']='/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/figures/moving_window'
#clustering
affinity_propagation_info=0
min_point=10 #for DBSCAN and OPTICS
eps_db=0.01

data=getDigitFeatures(feat_info)
time_data = data['time_data']
FDD_bc = data['FDD_bc']
label= data['label']

t_window=1 #how much time in a window
t_endFirst=time_data[0]+t_window
t_intv=np.where((time_data-t_endFirst)<0)[0][-1]
t_intv=len(FDD_bc)-1

# i=0
# j=t_intv

start_win=0
i=start_win
j=t_intv+start_win
max_dend_distance=np.zeros(len(label))
# k=1

# plt.plot(time_data,label)
# plt.show()
for k in range(start_win,len(label)-t_intv):#range(0,len(label[0::t_intv])):
    FDD_bc_win=FDD_bc[i:j,:]
    label_win=label[i:j,:]
    time_data_win=time_data[i:j]
    if label_win[0]==2:
        break
    # # normalize dataset for easier parameter selection
    # X = StandardScaler().fit_transform(FDD_bc_win) #to see whether there's a nan element np.any(np.isnan(mat))
    t = MinMaxScaler()
    t.fit(FDD_bc)
    X = t.transform(FDD_bc_win)

    if run_pca:
        #transforming the data using the first 3 PCA components. Note can't use Y=np.array(pca.transform(X)) unless we add back in the mean of each column of L  that was subtracted using
        Y = np.matmul( X- np.mean(X,axis=0), V)
        n_clust=1
        eps_db=0.7
        plot_info['axis_label']=["V1","V2","V3"]
        plot_info['plot_name']='PCA'
    else:
        n_clust=1
        eps_db=0.7
        plot_info['axis_label']=["E1","E2","E3"]
        plot_info['plot_name']='Encoder'
        if not allfeatModel:
            if forceAxis=='y':
                encoder=load_model('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/encoder_minmax_y_100.h5',compile=False)
            elif forceAxis=='x':
                encoder=load_model('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/encoder_minmax_x_100.h5',compile=False) 
        else:
            encoder=load_model('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/encoder_minmax_all_base_100.h5',compile=False)

        Y=encoder.predict(X)

    if plot_pca:
        #info to graph
        plot_info['time_data']=time_data_win
        plot_info['label']=label_win
        plot_info['Y']=Y
        if run_pca:
            plot_info['save_file_pca']=plot_info['save_file']+'/pca_'+str(k)+'.png'
        else:
            plot_info['save_file_pca']=plot_info['save_file']+'/encoder_'+str(k)+'.png'
        plotPCA(plot_info)




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


    if dendogram:
        linked=shc.linkage(
            Y, method='complete',metric='cityblock')
        c, coph_dists = cophenet(linked, pdist(Y)) #returns matrix where each row is [merge_point or clust idx 1, merge_point or clust idx 2, dist b/w merged points or clusters, #of clsuters formed due to merge]
                                                   #see  https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/ for more details
                                                   
        if write_to_text:
            if run_pca:
                file1=open(plot_info['save_file']+'/window_pca_cluster.txt','a')
            else:
                file1=open(plot_info['save_file']+'/window_encoder_cluster.txt','a')
            file1.write('\nWindow '+str(k)+ '\ttime interval '+ str(time_data_win[0])+':'+str(time_data_win[-1])+ '\tLabel:' + str(label_win[0]) +' -> ' +str(label_win[-1]) +'\n'+
            'Hierarchical Clustering:' + '\tCluster -2: points in cluster=%d' %linked[-2,-1]+' dist=%f' %linked[-2,2] +
            '\tCluster -3: points in cluster =%d' %linked[-3,-1]+' dist=%f' %linked[-3,2]+
            '\tFinal Cluster: Max. Distance: %f' %linked[-1,2] + ' points in cluster=%d' %linked[-1,-1] + ' (diff in points in cluster=%d)' %(linked[-1,-1]-linked[-2,-1]-linked[-3,-1]) +
            '\t Cophenet Score: %f\n' %c )
            file1.close()
        max_dend_distance[k]=linked[-1,2]
        
        if plot_dendogram:
            print(c)
            dend = shc.dendrogram(linked)
            # plt.savefig(plot_info['save_file']+'/dendrogram_'+str(k)+'.png',  bbox_inches='tight')
            # plt.close()
            plt.show()



    # default_base = {
    #                 # 'quantile': .3,
    #                 'eps': eps_db,
    #                 'damping': .85,
    #                 'preference': int(pref),
    #                 'n_neighbors': 2,
    #                 'n_clusters': 1,
    #                 'min_samples': 20,
    #                 'xi': 0.25,
    #                 'min_cluster_size': 0.1}
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

 
    params = default_base.copy()

    # # estimate bandwidth for mean shift
    # bandwidth = cluster.estimate_bandwidth(Y, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
    Y, n_neighbors=params['n_neighbors'], include_self=False)

    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    ward = cluster.AgglomerativeClustering(
            n_clusters=params['n_clusters'], linkage='complete',
            connectivity=connectivity, affinity='manhattan')
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
    optics = cluster.OPTICS(min_samples=params['min_samples_optics'],
                            xi=params['xi'], metric='manhattan')


    dbscan = cluster.DBSCAN(eps=params['eps'],min_samples=params['min_samples_dbscan'], metric='manhattan')

    birch = cluster.Birch(n_clusters=params['n_clusters'])

    affinity_propagation = cluster.AffinityPropagation(
            damping=params['damping'], preference=params['preference'],max_iter=1000,verbose=True, random_state=1)

    # affinity_propagation = cluster.AffinityPropagation(
    #         damping=params['damping'],max_iter=1000,verbose=True)


    clustering_algorithms = (
        #('AffinityPropagation', affinity_propagation),
        # ('SpectralClustering', spectral),
        # ('Complete', ward),
        ('DBSCAN', dbscan),
        ('OPTICS', optics),
        # ('Birch', birch)
    )
    # clustering_algorithms = (
    #      ('AffinityPropagation', affinity_propagation),
    # )
    if plot_cluster:
        fig=plt.figure(figsize=(9 * 2 + 3, 16.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)

        plot_num = 1

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
        # print('Clustering Algorithm: %s' %name)
        # print('Estimated number of clusters: %d' % n_clusters_)
        #print('Estimated number of noise points: %d' % n_noise_)
        if write_to_text:
            if run_pca:
                file1=open(plot_info['save_file']+'/window_pca_cluster.txt','a')
            else:
                file1=open(plot_info['save_file']+'/window_encoder_cluster.txt','a')
            file1.write('Clustering Algorithm: %s' %name +
            '\tEstimated number of clusters: %d' % n_clusters_ + 
            '\tEstimated number of noise points: %d' % n_noise_)
            
            if n_clusters_ >1:
                #print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(Y, labels))
                file1.write("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(Y, labels)+'\n')
            else:
                file1.write('\n')

            file1.close()
        if plot_cluster:     

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
            if n_clusters_ >1000:
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

        # plt.show()
    # i=i+t_intv
    # j=j+t_intv
    i=i+1
    j=j+1
    plt.close()
plt.scatter(np.arange(0,k),max_dend_distance[0:k])
plt.show()




