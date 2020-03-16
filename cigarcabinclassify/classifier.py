#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pandas as pd
import sklearn
import math
import os
import sys
import time

from tensorflow import keras
from tensorflow.keras.preprocessing import image

class CategoricalClassifier():

    layers = 4
    basepath = "/opt/cigarimg"
    batch_size = 32
    epochs = 40

    def __init__(self):
        pass

    def __init__(self, basepath, layers, batch_size, epochs):
        self.basepath = basepath
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs

    def predict(self, predict_dir):
        height = 128
        width = 128
        channels = 3
        init_filters = 16
        train_dir = self.basepath + "/training/training"

        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255,
            rotation_range = 20,
            shear_range = 0.2,
            zoom_range = 0.2,
            channel_shift_range=10,
            brightness_range=[0.1, 1],
            fill_mode = 'nearest',
        )

        train_generator = train_datagen.flow_from_directory(train_dir,
            target_size=(height, width),
            batch_size=self.batch_size,
            seed=7,
            shuffle=True,
            class_mode="categorical")

        num_classes = len(train_generator.class_indices)
        train_num = train_generator.samples

        model = keras.models.Sequential()
        for i in range(self.layers):
            filters = init_filters * 2**i
            if i==0:
                model.add(keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same',
                            activation='relu', input_shape=[width, height, channels]))
            else:
                model.add(keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same',
                            activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same',
                        activation='relu'))
            model.add(keras.layers.MaxPool2D(pool_size=2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(2**(self.layers-1)*init_filters, activation='relu'))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))

        model.compile(loss="categorical_crossentropy",
           optimizer="adam", metrics=['accuracy'])

        print(model.summary())
        
        model.fit_generator(train_generator,
           steps_per_epoch=train_num // self.batch_size,
           epochs= self.epochs)        

        model.save("cigar_model_1.h5")

        dir1 = os.listdir(predict_dir)
        dict1 = {}
        predict_list = []
        for file_name in dir1:
            img = image.load_img(predict_dir + "/" + file_name, target_size=(128, 128))
            img_arr = np.expand_dims(img, axis=0)
            preds = model.predict(img_arr/255)
            dict1[file_name] = np.argmax(preds)

        return dict1

class SparseClassifier():

    layers = 4
    basepath = "/opt/cigarimg"
    batch_size = 32
    epochs = 40

    def __init__(self):
        pass

    def __init__(self, basepath, layers, batch_size, epochs):
        self.basepath = basepath
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs

    def predict(self, predict_dir):
        height = 128
        width = 128
        channels = 3
        init_filters = 16
        train_dir = self.basepath + "/training/training"

        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255,
            rotation_range = 20,
            shear_range = 0.2,
            zoom_range = 0.2,
            channel_shift_range=10,
            brightness_range=[0.1, 1],
            fill_mode = 'nearest',
        )

        train_generator = train_datagen.flow_from_directory(train_dir,
            target_size=(height, width),
            batch_size=self.batch_size,
            seed=7,
            shuffle=True,
            class_mode="sparse")

        num_classes = len(train_generator.class_indices)
        train_num = train_generator.samples

        model = keras.models.Sequential()
        for i in range(self.layers):
            filters = init_filters * 2**i
            if i==0:
                model.add(keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same',
                            activation='relu', input_shape=[width, height, channels]))
            else:
                model.add(keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same',
                            activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same',
                        activation='relu'))
            model.add(keras.layers.MaxPool2D(pool_size=2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(2**(self.layers-1)*init_filters, activation='relu'))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))

        model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam", metrics=['accuracy'])

        print(model.summary())

        model.fit_generator(train_generator,
                                      steps_per_epoch=train_num // self.batch_size,
                                      epochs= self.epochs)

        model.save("cigar_model_2.h5")

        dir1 = os.listdir(predict_dir)
        dict1 = {}
        predict_list = []
        for file_name in dir1:
            img = image.load_img(predict_dir + "/" + file_name, target_size=(128, 128))
            img_arr = np.expand_dims(img, axis=0)
            preds = model.predict(img_arr/255)
            dict1[file_name] = np.argmax(preds)

        return dict1

class ResnetClassifier():

    layers = 4
    basepath = "/opt/cigarimg"
    batch_size = 32
    epochs = 40

    def __init__(self):
        pass

    def __init__(self, basepath, layers, batch_size, epochs):
        self.basepath = basepath
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs

    def predict(self, predict_dir):
        height = 128
        width = 128
        channels = 3
        init_filters = 16
        train_dir = self.basepath + "/training/training"

        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255,
            rotation_range = 20,
            shear_range = 0.2,
            zoom_range = 0.2,
            channel_shift_range=10,
            brightness_range=[0.1, 1],
            fill_mode = 'nearest',
        )

        train_generator = train_datagen.flow_from_directory(train_dir,
            target_size=(height, width),
            batch_size=self.batch_size,
            seed=7,
            shuffle=True,
            class_mode="sparse")

        num_classes = len(train_generator.class_indices)
        train_num = train_generator.samples

        model = keras.models.Sequential()
        for i in range(self.layers):
            filters = init_filters * 2**i
            if i==0:
                model.add(keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same',
                            activation='relu', input_shape=[width, height, channels]))
            else:
                model.add(keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same',
                            activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same',
                        activation='relu'))
            model.add(keras.layers.MaxPool2D(pool_size=2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(2**(self.layers-1)*init_filters, activation='relu'))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))

        model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam", metrics=['accuracy'])

        print(model.summary())

        model.fit_generator(train_generator,
                                      steps_per_epoch=train_num // self.batch_size,
                                      epochs= self.epochs)

        model.save("cigar_model_3.h5")

        dir1 = os.listdir(predict_dir)
        dict1 = {}
        predict_list = []
        for file_name in dir1:
            img = image.load_img(predict_dir + "/" + file_name, target_size=(128, 128))
            img_arr = np.expand_dims(img, axis=0)
            preds = model.predict(img_arr/255)
            dict1[file_name] = np.argmax(preds)

        return dict1

def main():
    cc = CategoricalClassifier("D:/ai_images", 2, 32, 1)
    test_predict = cc.predict("D:/ai_images/validation/validation")
    print(test_predict)

if __name__ == "__main__":
    main()