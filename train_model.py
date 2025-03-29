import os
import h5py
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mimg

from pathlib import Path
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
import random
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras import activations

import model as md
import util
import visualise
from parameters import *

import logging

logger = logging.getLogger('train_model')
logging.basicConfig(level=logging.INFO)


util.setSeed(seed_value)
data = util.readData(r'.')

train_data, val_data, test_data = util.splitTrainValTest(data, train_cutoff, val_cutoff)
class_weight = util.getClassWeights(train_data)

model = md.resnet50(
    (input_shape, input_shape, 3),
    l2_weight_decay,
    batch_norm_decay,
    batch_norm_epsilon,
    dropout_value,
    seed_value,
)
opt = Adam(lr=lr, decay=lr_decay)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, rotation_range=15, horizontal_flip=True, brightness_range=(0.9, 1.2)
)
test_val_datagen = ImageDataGenerator(rescale=1.0 / 255)


train_data_gen = train_datagen.flow_from_dataframe(
    train_data,
    shuffle=True,
    class_mode='binary',
    target_size=(input_shape, input_shape),
    x_col='image',
    y_col='label',
    batch_size=batch_size,
    validate_filename=False,
    seed=seed_value,
)

train_data_gen_unaug = test_val_datagen.flow_from_dataframe(
    train_data,
    class_mode='binary',
    target_size=(input_shape, input_shape),
    x_col='image',
    y_col='label',
    batch_size=batch_size,
    validate_filename=False,
    seed=seed_value,
)
val_data_gen = test_val_datagen.flow_from_dataframe(
    val_data,
    shuffle=False,
    class_mode='binary',
    target_size=(input_shape, input_shape),
    x_col='image',
    y_col='label',
    batch_size=batch_size,
    validate_filename=False,
    seed=seed_value,
)

test_data_gen = test_val_datagen.flow_from_dataframe(
    test_data,
    shuffle=False,
    class_mode='binary',
    target_size=(input_shape, input_shape),
    x_col='image',
    y_col='label',
    batch_size=batch_size,
    validate_filename=False,
    seed=seed_value,
)


if len(old_model_filename) == 0:
    if num_layers_copy > 0:
        model = md.transfer_layers(model, num_layers_copy, trainable=False)

    nb_train_steps = train_data.shape[0] // batch_size

    history = model.fit_generator(
        train_data_gen,
        class_weight=class_weight,
        epochs=nb_epochs,
        steps_per_epoch=nb_train_steps,
        validation_data=val_data_gen,
        validation_steps=val_data.shape[0] // batch_size,
    )

    if nb_epoch_unaug > 0:
        history2 = model.fit_generator(
            train_data_gen_unaug,
            class_weight=class_weight,
            epochs=nb_epoch_unaug,
            steps_per_epoch=nb_train_steps,
            validation_data=val_data_gen,
            validation_steps=val_data.shape[0] // batch_size,
        )
    visualise.generateHistoryPlots(history)
    model.save_weights(model_filemane)
else:
    model.load_weights(old_model_filename)

logger.info(model.evaluate_generator(train_data_gen_unaug, train_data.shape[0] // batch_size)[1])
logger.info(model.evaluate_generator(val_data_gen, val_data.shape[0] // batch_size)[1])
logger.info(model.evaluate_generator(test_data_gen, test_data.shape[0] // batch_size)[1])
preds = model.predict_generator(test_data_gen)

real_labels = test_data['label'].astype(int)
predicted_labels = (preds.reshape([-1]) > 0.5).astype(int)
visualise.plot_confusion_matrix(real_labels, predicted_labels, 'confusion_matrix.png')
files = test_data[test_data['label'] == '1'].sample(n=4).reset_index(drop=True)['image']
visualise.integratedGradients(model, files, (input_shape, input_shape))
