import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def RNN_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, stateful=True, batch_input_shape=(16, 10, 256)))
    model.add(LSTM(128, return_sequences=True, stateful=True))
    model.add(LSTM(128, stateful=True))
    model.add(Dense(4, activation='softmax'))
    return model