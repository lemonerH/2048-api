from models.Final_Model import RCNN_model
from models.get_data import data_generator 
from tensorflow.keras.callbacks import ModelCheckpoint
import os

model = RCNN_model()
filepath = "checkpoints/checkpoint.hdf5"
generator = data_generator(1000)

model.summary()

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath, monitor="val_acc", save_best_only = True, verbose=1, mode="auto")

if os.path.exists(filepath):
    model.load_weights(filepath)
    print("checkpoint_loaded")

model.fit_generator(generator, steps_per_epoch = 100, epochs = 50, callbacks = [checkpoint])
