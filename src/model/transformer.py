import numpy as np
import math
from numpy.random import normal
from typing import Optional, Union, Callable

import tensorflow as tf
tf.compat.v1.experimental.output_all_intermediates(True)

from tensorflow.keras import layers
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, LeakyReLU, LSTM, BatchNormalization
from keras.layers import Dense, Input, LSTM, Layer, TimeDistributed, Lambda, Activation, Add
from keras.layers import GlobalMaxPooling1D, Flatten, GlobalAveragePooling1D, Reshape, concatenate
from tensorflow.keras.initializers import Ones, Zeros
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras import callbacks
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# refer to # # https://www.kaggle.com/shujian/transformer-with-lstm
# define the multi-head transformer structure by hand
class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self._eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self._gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                      initializer=Ones(), trainable=True)
        self._beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                     initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self._gamma * (x - mean) / (std + self._eps) + self._beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self._temper = np.sqrt(d_model)
        self._dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self._temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+10) * (1 - x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self._dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, num_head, dim_model, dim_key, dim_value, dropout, mode=0, use_norm=True):
        self._head_mode = mode
        self._num_head = num_head
        self._dim_key = dim_key
        self._dim_value = dim_value
        self._dropout = dropout
        if self._head_mode == 0:
            self.qs_layer = Dense(num_head * dim_key, use_bias=False)
            self.ks_layer = Dense(num_head * dim_key, use_bias=False)
            self.vs_layer = Dense(num_head * dim_value, use_bias=False)
        elif self._head_mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(num_head):
                self.qs_layers.append(TimeDistributed(Dense(dim_key, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(dim_key, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(dim_value, use_bias=False)))
        self.attention = ScaledDotProductAttention(dim_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(dim_model))

    def __call__(self, q, k, v, mask=None):
        from keras.layers import Concatenate
        if self._head_mode == 0:
            qs = self.qs_layer(q)
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                shape = tf.shape(x)
                x = tf.reshape(x, [shape[0], shape[1], self._num_head, self._dim_key])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, shape[1], self._dim_key])
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, self._num_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                shape = tf.shape(x)
                x = tf.reshape(x, [self._num_head, -1, shape[1], shape[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, shape[1], self._num_head * self._dim_value])
                return x

            head = Lambda(reshape2)(head)
        elif self._head_mode == 1:
            heads = [];
            attns = []
            for i in range(self._num_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head);
                attns.append(attn)
            head = Concatenate()(heads) if self._num_head > 1 else heads[0]
            attn = Concatenate()(attns) if self._num_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self._dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        # outputs = Add()([outputs, q]) # sl: fix
        return self.layer_norm(outputs), attn


class Transformer:
    def __init__(self, win_len, input_dim, hidden_dim=[128, 64], attn_mode=0, head_mode=0):
        self.name = 'Transformer'
        self._win_len = win_len
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        # if attn_mode = 1, it will deploy the handmade multihead attention layer
        # if attn_mode = 0, it will deploy the keras multihead attention layer
        self._attn_mode = attn_mode
        self._head_mode = head_mode

    def build_and_compile(self) -> None:
        inputs = Input(shape=(self._win_len, self._input_dim))
        lstm_out1 = LSTM(self._hidden_dim[0], return_sequences=True)(inputs)
        x = LSTM(self._hidden_dim[1], return_sequences=True)(lstm_out1)
        if self._attn_mode == 1:
            x, slf_attn = MultiHeadAttention(num_head=3, dim_model=300,
                                             dim_key=self._hidden_dim[1],
                                             dim_value=self._hidden_dim[1],
                                             dropout=0.1,
                                             mode=self._head_mode)(x, x, x)
        elif self._attn_mode == 0:
            x, slf_attn = layers.MultiHeadAttention(
                num_heads=3,
                key_dim=self._hidden_dim[1],
                value_dim=self._hidden_dim[1],
                dropout=0.1)(x, x, x,
                             return_attention_scores=True)
        else:
            print('Wrong attn_mode input. Attention layer will not be deployed.')
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        dense = Dense(self._hidden_dim[1], activation="relu")(conc)
        dense_out = Dense(self._input_dim)(dense)

        model = Model(inputs=inputs, outputs=dense_out)

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