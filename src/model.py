import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Embedding, Dense, Dropout

class Attention_layer(Layer):
    '''
    # input shape:  [None, n, k]
    # output shape: [None, k]
    '''
    def __init__(self):
        super().__init__()

    def build(self, input_shape): # [None, field, k]
        self.attention_w = Dense(input_shape[1], activation='relu')
        self.attention_h = Dense(1, activation=None)

    def call(self, inputs, **kwargs): # [None, field, k]
        #if K.ndim(inputs) != 3:
        #    raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))
        # print(K.ndim(inputs))
        x = self.attention_w(inputs)  # [None, field, field]
        x = self.attention_h(x)       # [None, field, 1]
        a_score = tf.nn.softmax(x)
        a_score = tf.transpose(a_score, [0, 2, 1]) # [None, 1, field]
        output = tf.reshape(tf.matmul(a_score, inputs), shape=(-1, inputs.shape[2]))  # (None, k)
        return output      