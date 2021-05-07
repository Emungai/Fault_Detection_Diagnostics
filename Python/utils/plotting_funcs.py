import matplotlib.pyplot as plt
import math
#importing things for color map see https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.pyplot import clim
from matplotlib import ticker

def plotPCA(plot_info):
    #get variables
    #graphing plot wrt
    time_plot=plot_info['time_plot']
    escapetime_plot=plot_info['escapetime_plot']
    #info to graph
    time_data=plot_info['time_data']
    label=plot_info['label']
    Y=plot_info['Y']
    #color pallete of plot
    jet_cl=plot_info['jet_cl']

    axis_label=plot_info['axis_label']
    plot_name=plot_info['plot_name']
    
    fig=plt.figure(figsize=(9 * 2 + 3, 16.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)

    # fig = plt.figure()
    # fig1=plt.figure()
    # fig2=plt.figure()
    # fig3=plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax1 = fig1.add_subplot()
    # ax2 = fig2.add_subplot()
    # ax3 = fig3.add_subplot()

    ax = fig.add_subplot(1,4,1,projection='3d')
    ax1 = fig.add_subplot(1,4,2)
    ax2 = fig.add_subplot(1,4,3)
    ax3 = fig.add_subplot(1,4,4)
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
    
        
    V1=axis_label[0]
    V2=axis_label[1]
    V3=axis_label[2]

    ax.set_xlabel(V1)
    ax.set_ylabel(V2)
    ax.set_zlabel(V3)
    ax.set_title(plot_name)
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

    ax1.set_xlabel(V1)
    ax1.set_ylabel(V2)
 

    ax2.set_xlabel(V1)
    ax2.set_ylabel(V3)

    ax3.set_xlabel(V2)
    ax3.set_ylabel(V3)

    plt.show()
    # plt.savefig(plot_info['save_file_pca'],  bbox_inches='tight')
    # plt.close(fig)