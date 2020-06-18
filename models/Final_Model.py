import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.layers import Flatten, Dropout, Conv2D, TimeDistributed
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate


def RCNN_model():
    inputs = Input((4, 8))

    lstm1 = LSTM(256, return_sequences = True, activation = 'relu')(inputs)
    lstm2 = LSTM(256, return_sequences = True, activation = 'relu')(lstm1)
    lstm3 = LSTM(256, return_sequences = True, activation = 'relu')(lstm2)
    lstm4 = LSTM(256, return_sequences = False, activation = 'relu')(lstm3)

    flatten1 = Flatten()(lstm4)
    de1 = Dense(256, activation = 'relu')(flatten1)

    reshape1 = Reshape((4, 8, 1))(inputs)
    conv1 = Conv2D(filters = 1, kernel_size = 2, strides = 2)(reshape1)
    
    flatten2 = Flatten()(conv1)
    de2 = Dense(256, activation = 'relu')(flatten2)
    # reshape2 = Reshape((8, 256))(de2)

    hidden = concatenate([de1, de2])
    flatten3 = Flatten()(hidden)

    outputs = Dense(4, activation = 'softmax')(flatten3)
    model = Model(inputs, outputs)
    return model