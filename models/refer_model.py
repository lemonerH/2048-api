import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Activation, concatenate

def refer_model():
		inputs = Input((4, 4, 16))

		conv = inputs
		FLITERS = 128
		conv41 = Conv2D(filters = FLITERS, kernel_size = (4, 1), kernel_initializer = 'he_uniform')(conv)
		conv14 = Conv2D(filters = FLITERS, kernel_size = (1, 4), kernel_initializer = 'he_uniform')(conv)
		conv22 = Conv2D(filters = FLITERS, kernel_size = (2, 2), kernel_initializer = 'he_uniform')(conv)
		conv33 = Conv2D(filters = FLITERS, kernel_size = (3, 3), kernel_initializer = 'he_uniform')(conv)
		conv44 = Conv2D(filters = FLITERS, kernel_size = (4, 4), kernel_initializer = 'he_uniform')(conv)

		hidden = concatenate([Flatten()(conv41), Flatten()(conv14), Flatten()(conv22), Flatten()(conv33), Flatten()(conv44)])
		x = BatchNormalization()(hidden)
		x = Activation('relu')(hidden)

		for width in [512, 128]:
			    x = Dense(width, kernel_initializer = 'he_uniform')(x)
			    x = BatchNormalization()(x)
			    x = Activation('relu')(x)

		outputs = Dense(4, activation = 'softmax')(x)
		model = Model(inputs, outputs)
		return model