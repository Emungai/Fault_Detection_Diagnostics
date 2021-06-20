import numpy as np
from pandas import concat
from pandas import DataFrame
import math
import tensorflow as tf

def get_LSTMData(raw,window_size,shift,num_features):
    # data=np.concatenate((np.arange(10),np.arange(50,60))).reshape(10,2)
    # raw=data
    # values=raw.values
    input=raw.shift(1)
    label=raw.shift(-1)
    input.dropna(inplace=True)
    label.dropna(inplace=True)
    # window_size=4
    # shift=2
    input_values=input.values
    label_values=label.values
    num_window_total=math.floor((input_values.shape[0]-window_size+shift)/shift)
    j=0
    k=window_size
    win_stacked=np.array([np.zeros((window_size,num_features))])
    label_stacked=np.array([np.zeros((window_size,num_features))])
    for i in range(num_window_total):
        window=np.array([input_values[j:k,]])
        label_win=np.array([label_values[j:k,]])
        win_stacked=tf.concat((win_stacked,window),axis=0)
        label_stacked=tf.concat((label_stacked,label_win),axis=0)
        j+=shift
        k+=shift
    win_stacked=win_stacked[1::,:,:]
    label_stacked=label_stacked[1::,:,:]
    return win_stacked,label_stacked

