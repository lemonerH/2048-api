import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Flatten, Dropout, Conv2D, TimeDistributed
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate


def CRNN_model():
		inputs = Input((None, 4, 4, 16))

		conv = inputs
		FLITERS = 128
		conv41 = TimeDistributed(Conv2D(filters = FLITERS, kernel_size = (4, 1), kernel_initializer = 'he_uniform'))(conv)
		conv14 = TimeDistributed(Conv2D(filters = FLITERS, kernel_size = (1, 4), kernel_initializer = 'he_uniform'))(conv)
		conv22 = TimeDistributed(Conv2D(filters = FLITERS, kernel_size = (2, 2), kernel_initializer = 'he_uniform'))(conv)
		conv33 = TimeDistributed(Conv2D(filters = FLITERS, kernel_size = (3, 3), kernel_initializer = 'he_uniform'))(conv)
		conv44 = TimeDistributed(Conv2D(filters = FLITERS, kernel_size = (4, 4), kernel_initializer = 'he_uniform'))(conv)

		flatten41 = TimeDistributed(Flatten())(conv41)
		flatten14 = TimeDistributed(Flatten())(conv14)
		flatten22 = TimeDistributed(Flatten())(conv22)
		flatten33 = TimeDistributed(Flatten())(conv33)
		flatten44 = TimeDistributed(Flatten())(conv44)
		
		hidden = concatenate([flatten41, flatten14, flatten22, flatten33, flatten44])
		
		lstm = LSTM(50)(hidden)

		outputs = Dense(4, activation = 'softmax')(lstm)
		model = Model(inputs, outputs)
		return model