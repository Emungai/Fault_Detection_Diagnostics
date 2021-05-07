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

# train autoencoder for classification with no compression in the bottleneck layer
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot


from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model


#appending a path
sys.path.append('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/utils')


from similarity_mat import sim_matrix
from LoadFeatures import getDigitFeatures
from plotting_funcs import plotPCA


#VARIABLES

load_singleFile=0
feat_info={}
feat_info['diff_sampleRate']=0
feat_info['all_feat']=1
#loading dataa
if load_singleFile:
    forceAxis="x"
    feat_info['forceAxis']=forceAxis
    feat_info['file_name']='100N_4-10-21.mat'
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




getEncoderModel=1

X = FDD_bc #to see whether there's a nan element np.any(np.isnan(mat))
y=label

if getEncoderModel:
    n_inputs = X.shape[1]

    # split into train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # scale data
    # X_train=StandardScaler().fit_transform(X_train)
    # X_test=StandardScaler().fit_transform(X_test)

    # scale data
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)

    # define encoder
    visible = Input(shape=(n_inputs,))
    # encoder level 1
    e = Dense(n_inputs*2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # encoder level 2
    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # bottleneck
    # n_bottleneck = round(float(n_inputs) / 2.0)
    n_bottleneck = 3
    bottleneck = Dense(n_bottleneck)(e)

    # define decoder, level 1
    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # decoder level 2
    d = Dense(n_inputs*2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)

    # output layer
    output = Dense(n_inputs, activation='linear')(d)

    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')
    # plot the autoencoder
    plot_model(model, 'autoencoder_compress.png', show_shapes=True)
    # fit the autoencoder model to reconstruct input
    history = model.fit(X_train, X_train, epochs=200, batch_size=16, verbose=2, validation_data=(X_test,X_test))

    # plot loss
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # pyplot.show()
    # define an encoder model (without the decoder)
    encoder = Model(inputs=visible, outputs=bottleneck)
    plot_model(encoder, 'encoder_no_compress.png', show_shapes=True)
    # save the encoder to file
    if not feat_info["all_feat"]:
        if forceAxis=="y":
            encoder.save('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/encoder_minmax_y_100.h5')
        elif forceAxis=="x":
            encoder.save('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/encoder_minmax_x_100.h5')
    else:
        encoder.save('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/encoder_minmax_all_base_100.h5')

else:
    if not feat_info["all_feat"]:
        if forceAxis=="y":
            encoder=load_model('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/encoder_minmax_y_100.h5',compile=False)
        elif forceAxis=="x":
            encoder=load_model('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/encoder_minmax_x_100.h5',compile=False) 
    else:
        encoder=load_model('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/encoder_minmax_all_base_100.h5')
    

# X = StandardScaler().fit_transform(FDD_bc) #to see whether there's a nan element np.any(np.isnan(mat))

t = MinMaxScaler()
t.fit(FDD_bc)
X = t.transform(FDD_bc)


Y=encoder.predict(X)


#info to graph
#graphing plot wrt
plot_info={}
plot_info['time_plot']=0
plot_info['escapetime_plot']=1

#color pallete of plot
plot_info['jet_cl']=1 #if true, will plot using jet color map otherwise, we'll use viradis
plot_info['time_data']=time_data
plot_info['label']=label
plot_info['Y']=Y
plot_info['axis_label']=["E1","E2","E3"]
plot_info['plot_name']='Encoder'

plotPCA(plot_info)