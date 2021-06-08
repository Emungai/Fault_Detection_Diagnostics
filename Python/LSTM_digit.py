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

# #appending a path
sys.path.append('/home/exo/Documents/eva/Fault_Detection_Diagnostics/Python/utils')

from LoadFeatures import getDigitData
from LoadFeatures import getDigitState
from WindowGenerator import WindowGenerator  

#loading data
forceAxis="x"
feat_info={}
feat_info['diff_sampleRate']=1
feat_info['all_feat']=1
 


#loading dataa
feat_info['forceAxis']=forceAxis
# feat_info['file_name']='100N_4-10-21.mat'
# feat_info['file_name']='-100N_4-25-21.mat'
# feat_info['file_name']='100N_noise_5-6-21'
feat_info['file_name']='70N_5-6-21'
X,time,u =getDigitState(feat_info)
data=np.column_stack((X,u))
#data preparation
# data= np.transpose(np.row_stack((q_all_f, dq_all_f,ua_all_f)))
n = len(data)
num_features = data.shape[1]
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

#standardizing data
# scale data
t = MinMaxScaler()
t.fit(train_data)
train_data = t.transform(train_data)
test_data = t.transform(test_data)
val_data=t.transform(val_data)
#converting to table format
train_data_tbl=pd.DataFrame(train_data)
test_data_tbl=pd.DataFrame(test_data)
val_data_tbl=pd.DataFrame(val_data)


w1 = WindowGenerator(input_width=100, label_width=100,shift=1,train_df=train_data_tbl,val_df=val_data_tbl,test_df=test_data_tbl)

MAX_EPOCHS = 100

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history


lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(50, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=num_features)
])

history = compile_and_fit(lstm_model, w1)

IPython.display.clear_output()
val_performance = {}
performance = {}
val_performance['LSTM'] = lstm_model.evaluate( w1.val)
performance['LSTM'] = lstm_model.evaluate( w1.test, verbose=0)
