import os
import datetime
import sys

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.saving.save import load_model

# #appending a path
sys.path.append('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/utils')

from LoadFeatures import getDigitFeatures
from LoadFeatures import getDigitState
from WindowGenerator import WindowGenerator  
from format_data import get_LSTMData

train_model=0
#loading data
forceAxis="x"
feat_info={}
feat_info['diff_sampleRate']=1
feat_info['all_feat']=0
 


#loading dataa
feat_info['forceAxis']=forceAxis
# feat_info['file_name']='100N_4-10-21.mat'
# feat_info['file_name']='-100N_4-25-21.mat'
# feat_info['file_name']='100N_noise_5-6-21'
# feat_info['file_name']='70N_5-6-21'
LSTM_data={}
file_names=['0N_6-7-21','10N_6-7-21','25N_6-7-21','45N_6-7-21','50N_6-7-21','60N_6-7-21','65N_6-7-21','70N_5-6-21','75N_6-7-21','80N_6-7-21','90N_6-7-21','100N_4-10-21','200N_6-7-21','5N_6-7-21','15N_6-7-21','20N_6-7-21','30N_6-7-21','35N_6-7-21','40N_6-7-21','55N_6-7-21','85N_6-7-21','95N_6-7-21','105N_6-7-21','110N_6-7-21','115N_6-7-21','120N_6-7-21','125N_6-7-21','130N_6-7-21']
for file in file_names:
    feat_info['file_name']=file
    # X,time,u =getDigitState(feat_info)
    FDD_data, X,time,u=getDigitFeatures(feat_info)
    data=np.column_stack((X,u))
    #data preparation
    # data= np.transpose(np.row_stack((q_all_f, dq_all_f,ua_all_f)))
    n = len(data)
    num_features = data.shape[1]
    # train_data = data[0:int(n*0.7)]
    # val_data = data[int(n*0.7):int(n*0.9)]
    # test_data = data[int(n*0.9):]

    #standardizing data
    # scale data
    t = MinMaxScaler()
    t.fit(data)
    # train_data = t.transform(train_data)
    # test_data = t.transform(test_data)
    # val_data=t.transform(val_data)
    #converting to table format
    data_tbl=pd.DataFrame(data)
    window_size=100
    shift=1
    X_file, y_file= get_LSTMData(data_tbl,window_size,shift,num_features)
    LSTM_data[file]=X_file,y_file,time
X_train=np.concatenate((LSTM_data[file_names[0]][0],LSTM_data[file_names[2]][0],LSTM_data[file_names[4]][0],LSTM_data[file_names[5]][0],LSTM_data[file_names[7]][0],LSTM_data[file_names[9]][0],LSTM_data[file_names[11]][0], LSTM_data[file_names[13]][0], LSTM_data[file_names[14]][0],LSTM_data[file_names[15]][0], LSTM_data[file_names[16]][0], LSTM_data[file_names[17]][0], LSTM_data[file_names[18]][0], LSTM_data[file_names[19]][0], LSTM_data[file_names[20]][0], LSTM_data[file_names[21]][0], LSTM_data[file_names[22]][0], LSTM_data[file_names[24]][0], LSTM_data[file_names[25]][0]    ))
y_train=np.concatenate((LSTM_data[file_names[0]][1],LSTM_data[file_names[2]][1],LSTM_data[file_names[4]][1],LSTM_data[file_names[5]][1],LSTM_data[file_names[7]][1],LSTM_data[file_names[9]][1],LSTM_data[file_names[11]][1], LSTM_data[file_names[13]][1], LSTM_data[file_names[14]][1],LSTM_data[file_names[15]][1], LSTM_data[file_names[16]][1], LSTM_data[file_names[17]][1], LSTM_data[file_names[18]][1], LSTM_data[file_names[19]][1], LSTM_data[file_names[20]][1], LSTM_data[file_names[21]][1], LSTM_data[file_names[22]][1], LSTM_data[file_names[24]][1], LSTM_data[file_names[25]][1]))

X_val=np.concatenate((LSTM_data[file_names[3]][0],LSTM_data[file_names[6]][0],LSTM_data[file_names[8]][0], LSTM_data[file_names[23]][0]))
y_val=np.concatenate((LSTM_data[file_names[3]][1],LSTM_data[file_names[6]][1],LSTM_data[file_names[8]][1], LSTM_data[file_names[23]][0]))

X_test=np.concatenate((LSTM_data[file_names[1]][0],LSTM_data[file_names[10]][0],LSTM_data[file_names[12]][0]))
y_test=np.concatenate((LSTM_data[file_names[1]][1],LSTM_data[file_names[10]][1],LSTM_data[file_names[12]][1]))

# w1 = WindowGenerator(input_width=100, label_width=100,shift=1,train_df=train_data_tbl,val_df=val_data_tbl,test_df=test_data_tbl)
if train_model:
    MAX_EPOCHS = 300
    patience=2



    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(50, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=num_features)
    ])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    file_path='/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/lstm_model_pos_x_best.h5'
    model_checkpoint=tf.keras.callbacks.ModelCheckpoint(file_path,monitor='val_loss',save_best_only=True)

    lstm_model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
    # lstm_model.compile(loss=tf.losses.MeanSquaredError(),
    #         optimizer=tf.optimizers.Adam(),
    #         metrics=[tf.metrics.Accuracy()])

    history = lstm_model.fit(X_train,y_train, epochs=MAX_EPOCHS,batch_size=40,
                        validation_data=(X_val,y_val),
                        callbacks=[early_stopping,model_checkpoint])
    # history = lstm_model.fit(X_train,y_train, epochs=MAX_EPOCHS,batch_size=40,
    #                     validation_data=(X_val,y_val))


    lstm_model.save('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/lstm_model_pos_x_FIXED_Longer_xfeatures_more_0-1sApart.h5')
else:
    lstm_model_morebw=tf.keras.models.load_model('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/lstm_model_pos_x_FIXED_Longer_xfeatures_more.h5')
    lstm_model_x=tf.keras.models.load_model('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/lstm_model_pos_x_FIXED_Longer_xfeatures.h5')
    lstm_model_all=tf.keras.models.load_model('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/lstm_model_pos_x_FIXED.h5')
    lstm_model_morebw_nSpaced=tf.keras.models.load_model('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/model/lstm_model_pos_x_FIXED_Longer_xfeatures_more.h5')

lstm_model=lstm_model_x

num=12
y_predict=lstm_model.predict(LSTM_data[file_names[num]][0])
y_test=LSTM_data[file_names[num]][1]
y_diff=y_test-y_predict
y_diff_states=y_diff[:,:,1:X.shape[1]]
np.amax(y_diff_states[:,99,:])
y_predlast=y_diff_states[:,99,:]
time_plt=LSTM_data[file_names[num]][2][100:]
max_list=[]
idx=[]
for j in range(len(y_predlast)):
    row=y_predlast[j,:]
    max_list.append(np.amax(row))
    idx.append(np.where(row == np.amax(row))[0][0])
max_max=np.amax(max_list)
idx_fin=np.where(max_list == np.amax(max_list))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(time_plt,idx,max_list)
ax.set_xlabel('time (s)')
ax.set_ylabel('index')
ax.set_zlabel('max value')
plt.show()
# y_predict=lstm_model.predict(X_test[0,:,].reshape(1,100,76))

# #getting max diff
# max_diff=[]
# diff_all=[]
# idx_max=[]
# y_predict=lstm_model.predict(LSTM_data[file_names[0]][0])
# time_plt=LSTM_data[file_names[1]][2][99:]
# diff_file=np.abs(tf.reshape(LSTM_data[file_names[0]][1]-y_predict,[time_plt.shape[0],-1]))

# for j in range(num_features):
#     diff_all.append(np.amax(diff_file[:,j]))
# max_diff.append(np.amax(diff_all))
# idx_max.append(np.argmax(diff_all))

# y_predict=lstm_model.predict(LSTM_data[file_names[10]][0])
# time_plt=LSTM_data[file_names[10]][2][99:]
# diff_file=np.abs(tf.reshape(LSTM_data[file_names[10]][1]-y_predict,[time_plt.shape[0],-1]))

# for j in range(num_features):
#     diff_all.append(np.amax(diff_file[:,j]))
# max_diff.append(np.amax(diff_all))
# idx_max.append(np.argmax(diff_all))


# y_predict=lstm_model.predict(LSTM_data[file_names[12]][0])
# time_plt=LSTM_data[file_names[12]][2][99:]
# diff_file=np.abs(tf.reshape(LSTM_data[file_names[12]][1]-y_predict,[time_plt.shape[0],-1]))

# for j in range(num_features):
#     diff_all.append(np.amax(diff_file[:,j]))
# max_diff.append(np.amax(diff_all))
# idx_max.append(np.argmax(diff_all))


# #plotting results
# fig=plt.figure(figsize=(9 * 2 + 3, 16.5))
# plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
#                             hspace=.01)

# # ax1 = fig.add_subplot(1,4,1)
# # ax2 = fig.add_subplot(1,4,2)
# # ax3 = fig.add_subplot(1,4,3)
# plt.subplot(3, 1, 1)
# y_predict=lstm_model.predict(LSTM_data[file_names[1]][0])
# y_predict=tf.reshape(y_predict,[-1,num_features])
# y_actual=LSTM_data[file_names[1]][1]
# # y_actual=tf.reshape(,[-1,num_features]).shape

# time_plt=LSTM_data[file_names[1]][2][99:]
# plt.plot(time_plt,np.abs(tf.reshape(LSTM_data[file_names[1]][1]-y_predict,[time_plt.shape[0],-1])))

# plt.subplot(3, 1, 2)
# y_predict=lstm_model.predict(LSTM_data[file_names[10]][0])
# time_plt=LSTM_data[file_names[10]][2][99:]
# plt.plot(time_plt,np.abs(tf.reshape(LSTM_data[file_names[10]][1]-y_predict,[time_plt.shape[0],-1])))

# plt.subplot(3, 1, 3)
# y_predict=lstm_model.predict(LSTM_data[file_names[12]][0])
# time_plt=LSTM_data[file_names[12]][2][99:]
# plt.plot(time_plt,np.abs(tf.reshape(LSTM_data[file_names[12]][1]-y_predict,[time_plt.shape[0],-1])))


# IPython.display.clear_output()
# val_performance = {}
# performance = {}
# val_performance['LSTM'] = lstm_model.evaluate( X_val,y_val)
# performance['LSTM'] = lstm_model.evaluate( X_test,y_test, verbose=0)
# train_performance['LSTM'] = lstm_model.evaluate( X_train,y_train)

# # max_n=2
# # plot_col_index=0
# # plt.figure(figsize=(12, 8))
# # for n in range(max_n):
# #     plt.subplot(max_n, 1, n+1)
# #     plt.ylabel('x1')
# #     plt.plot(time[0:100], X_file[n, :, plot_col_index],
# #                 label='Inputs', marker='.', zorder=-10)

   
# #     label_col_index = plot_col_index

# #     # if label_col_index is None:
# #     #     continue

# #     plt.scatter(time[1:101],y_file[n, :, label_col_index],
# #                 edgecolors='k', label='Labels', c='#2ca02c', s=64)
    
# #     predictions = lstm_model(X_file)
# #     plt.scatter(time[1:101], predictions[n, :, label_col_index],
# #                 marker='X', edgecolors='k', label='Predictions',
# #                 c='#ff7f0e', s=64)

# #     if n == 0:
# #         plt.legend()
# def plot(model=None, plot_col=1, max_subplots=5,y_label='x1',time=[]):
#   inputs, labels = self.example
#   plt.figure(figsize=(12, 8))
#   plot_col_index = self.column_indices[plot_col]
#   max_n = min(max_subplots, len(inputs))
#   for n in range(max_n):
#     plt.subplot(max_n, 1, n+1)
#     plt.ylabel('x1')
#     plt.plot(time[0:100], X_file[n, :, plot_col_index],
#                 label='Inputs', marker='.', zorder=-10)

   
#     label_col_index = plot_col_index

#     # if label_col_index is None:
#     #     continue

#     plt.scatter(time[1:101],y_file[n, :, label_col_index],
#                 edgecolors='k', label='Labels', c='#2ca02c', s=64)
    
#     predictions = lstm_model(X_file)
#     plt.scatter(time[1:101], predictions[n, :, label_col_index],
#                 marker='X', edgecolors='k', label='Predictions',
#                 c='#ff7f0e', s=64)

#     if n == 0:
#         plt.legend()

#   plt.xlabel('Time [s]')