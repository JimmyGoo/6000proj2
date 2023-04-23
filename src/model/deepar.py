import numpy as np
import math
from numpy.random import normal
from typing import Optional, Union, Callable

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras import callbacks
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

class GaussianLayer(Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        """Init."""
        self._input_dim = input_dim
        self._output_dim = output_dim
        # self._kernel_mu, self._kernel_sigma, self.bias_1, self.bias_2 = [], [], [], []
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build the weights and biases."""
        self._kernel_mu = self.add_weight(
            name="kernel_mu",
            shape=(self._input_dim, self._output_dim),
            initializer=glorot_normal(),
            trainable=True,
        )
        self._kernel_sigma = self.add_weight(
            name="kernel_sigma",
            shape=(self._input_dim, self._output_dim),
            initializer=glorot_normal(),
            trainable=True,
        )
        self._bias_mu = self.add_weight(
            name="bias_mu",
            shape=(self._output_dim,),
            initializer=glorot_normal(),
            trainable=True,
        )
        self._bias_sigma = self.add_weight(
            name="bias_sigma",
            shape=(self._output_dim,),
            initializer=glorot_normal(),
            trainable=True,
        )
        super(GaussianLayer, self).build(input_shape)

    def call(self, x):
        """Do the layer computation."""
        # print(x.shape)
        output_mu = K.dot(x, self._kernel_mu) + self._bias_mu
        output_sig = K.dot(x, self._kernel_sigma) + self._bias_sigma
        output_sig_pos = K.log(1 + K.exp(output_sig)) + 1e-06
        return [output_mu, output_sig_pos]

class DeepAR:
    def __init__(self, win_len, input_dim, hidden_dim=[256,128]):

        self._win_len = win_len
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

    def _build_loss(self, sigma: tf.Tensor) -> Callable:

        def gaussian_likelihood(y_true, y_pred):
            return tf.reduce_mean(
                tf.math.log(tf.math.sqrt(2 * math.pi))
                + tf.math.log(sigma)
                + tf.math.truediv(
                    tf.math.square(y_true - y_pred), 2 * tf.math.square(sigma)
                )
            )
            
        return gaussian_likelihood

    def build_and_compile(self) -> None:

        inputs = Input(shape=(self._win_len, self._input_dim))
        lstm_out = LSTM(
            self._hidden_dim[0],
            return_sequences=False,
            # dropout=0.1,
        )(inputs)
        dense_out = Dense(self._hidden_dim[1], activation='relu')(lstm_out)
        mu, sigma = GaussianLayer(self._hidden_dim[1], self._input_dim, name='gaussian_layer')(dense_out)

        model = Model(inputs, mu)

        model.compile(
            loss=self._build_loss(sigma),
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

        get_intermediate = K.function(
            inputs=[self._model.input],
            outputs=self._model.get_layer('gaussian_layer').output,
        )

        output = get_intermediate([X_test])
        y_pred = []
        for mu, sigma in zip(output[0].reshape(-1), output[1].reshape(-1)):
            sample = normal(
                loc=mu, scale=sigma, size=1
            )
            y_pred.append(sample)
            
        return np.array(y_pred).reshape((-1, self._input_dim))
