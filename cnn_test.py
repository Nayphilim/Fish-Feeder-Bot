# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 23:47:35 2020

@author: Nayphilim
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

imgHeight = 680
imgWidth = 488
batchSize = 4

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

model = keras.Sequential([
    layers.Input((imgHeight, imgWidth, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')

    ])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=(True)),
    optimizer = keras.optimizers.Adam(3e-4), 
    metrics=['accuracy'],
    )



dsTrain = tf.keras.preprocessing.image_dataset_from_directory(
    'creature_type_cnn',
    labels = 'inferred',
    label_mode = "int",
    color_mode = 'rgb',
    batch_size = batchSize,
    image_size = (imgHeight, imgWidth),
    shuffle = True,
    seed=123,
    validation_split=0.1,
    subset="training",
    )

dsVal = tf.keras.preprocessing.image_dataset_from_directory(
    'creature_type_cnn',
    labels = 'inferred',
    label_mode = "int",
    color_mode = 'rgb',
    batch_size = batchSize,
    image_size = (imgHeight, imgWidth),
    shuffle = True,
    seed=123,
    validation_split=0.1,
    subset="validation",
    )

print(model.summary())
try:
    with(tf.device('/GPU:0')):
        model.fit(dsTrain, batch_size=batchSize, epochs=10)
        model.save('creature_type_model_bgcd/')
except RuntimeError as e:
  print(e)