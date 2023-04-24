import numpy as np
import math
from numpy.random import normal

import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, LeakyReLU, Input, Layer
from tensorflow.keras.models import Model

# The model structure refers to https://www.kaggle.com/code/yzgast/cnn-1d-for-stock-prediction/notebook

class CNN1D:
    def __init__(self, win_len, input_dim, filter=[32, 64, 128], kernel_size=[1, 1, 1], dense=256):
        self._win_len = win_len
        self._input_dim = input_dim
        self._filter = filter
        self._kernel_size = kernel_size
        self._dense = dense

    def build_and_compile(self) -> None:
        inputs = Input(shape=(self._win_len, self._input_dim))
        conv1 = Conv1D(filters=self._filter[0],
                       kernel_size=self._kernel_size[0], padding='same',
                       activation='relu',
                       input_shape=(self._win_len, self._input_dim),
                       data_format="channels_last")(inputs)
        conv2 = Conv1D(filters=self._filter[1],
                       kernel_size=self._kernel_size[1], padding='same',
                       activation='relu', data_format="channels_last")(conv1)
        maxpool = MaxPooling1D(pool_size=(1))(conv2)
        conv3 = Conv1D(filters=self._filter[2],
                       kernel_size=self._kernel_size[2], padding='same',
                       activation='relu', data_format="channels_last")(maxpool)

        x = Flatten()(conv3)
        dense = Dense(self._dense)(x)
        dense = LeakyReLU(alpha=0.01)(dense)
        dropout = Dropout(0.8)(dense)
        dense_out = Dense(self._input_dim)(dropout)

        model = Model(inputs, dense_out)
        model.compile(
            loss='mse',
            metrics=['mse'],
        )

        self._model = model

    def get_model(self) -> Model:
        return self._model

    def summary(self) -> None:
        self._model.summary()

    def fit(self, **kwargs) -> None:
        self._model.fit(**kwargs)

    def predict(self, X_test):
        return self._model.predict(X_test)