from models.Final_Model import RCNN_model
from models.get_data import data_generator 
from tensorflow.keras.callbacks import ModelCheckpoint
import os

import csv
import numpy as np
import time

datas = []
labels = []

def step2array(step):
    vec = np.zeros(4, dtype = bool)
    vec[step] = 1
    return vec

start = time.time()
with open("DATA.csv", "r") as f:
    csv_read = csv.reader(f)
    for line in csv_read:
        board = np.zeros((4, 4))
        for i in range(16):
            board[i // 4][i % 4] = int(line[i]) / 11.0
        boardT = board.T
        step = int(line[16])
        datas.append(np.hstack((board, boardT)))
        labels.append(step2array(step))
print("time: ", time.time() - start)

datas = np.array(datas)
labels = np.array(labels)

print(datas.shape)
print(labels.shape)

model = RCNN_model()
filepath = "checkpoints/checkpoint.hdf5"

model.summary()

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath, monitor="val_acc", save_best_only = True, verbose=1, mode="auto")

if os.path.exists(filepath):
    model.load_weights(filepath)
    print("checkpoint_loaded")

model.fit(datas, labels, batch_size = 1000, epochs = 10, callbacks = [checkpoint])
