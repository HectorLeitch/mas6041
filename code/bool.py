import tensorflow as tf
import numpy as np
import itertools
import math
from matplotlib import pyplot as plt
import os


def make_u_model(u):
    """
    Create a model that evaluates a Boolean function defined by the input u.

    The first layer creates 2^n - 1 varaibles using the ReLU activation layer.
    The second layer combines them according to weights derived from u.
    Works reasonably quickly for n <= 8.
    """
    n = int(np.log2(u.shape[0]))
    inputs = tf.keras.Input(shape=(n))
    dense1 = tf.keras.layers.Dense(2**n-1, activation='relu')(inputs)
    dense2 = tf.keras.layers.Dense(1, activation='linear')(dense1)
    u_model = tf.keras.Model(inputs=inputs, outputs=dense2, name='u_model')
    u_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    bool_in = list(itertools.product(*[[0,1]]*n))

    w0 = np.array(bool_in)[1:]
    b0 = 1-sum(w0.T)
    u_model.layers[1].set_weights([w0.T,b0])

    w1 = np.zeros((1,2**n))
    for j in range(2**n):
        w1[:,j] = w1[:,j] + u[j]
        for k in range(j):
            if all(np.array(bool_in)[j] >= np.array(bool_in)[k]):
                w1[:,j] = w1[:,j] - w1[:,k]
    # drop first term u0 from the weight matrix and add it in as a bias
    w1 = w1[:,1:]
    b1 = np.array([u[0]])
    u_model.layers[2].set_weights([w1.T,b1])
    return u_model


def make_n_model(n):
    """
    Create a model to evaluate all Boolean functions simultaneously, given n.

    The first layer creates 2^n - 1 varaibles using the ReLU activation layer.
    The second layer combines them according to weights derived from the u
    values across all possible Boolean functions.
    Works for n < 5, after which there are too many weights.
    """
    inputs = tf.keras.Input(shape=(n))
    dense1 = tf.keras.layers.Dense(2**n-1, activation='relu')(inputs)
    dense2 = tf.keras.layers.Dense(2**2**n, activation='linear')(dense1)
    n_model = tf.keras.Model(inputs=inputs, outputs=dense2, name='n_model')
    n_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    bool_in = list(itertools.product(*[[0,1]]*n))
    bool_funs = list(itertools.product(*[[0,1]]*(2**n)))

    w0 = np.array(bool_in)[1:]
    b0 = 1-sum(w0.T)
    n_model.layers[1].set_weights([w0.T,b0])

    # same as first model but using columns of bool_funs instead of u
    w1 = np.zeros((2**2**n,2**n))
    for j in range(2**n):
        w1[:,j] = w1[:,j] + np.array(bool_funs)[:,j]
        for k in range(j):
            if all(np.array(bool_in)[j] >= np.array(bool_in)[k]):
                w1[:,j] = w1[:,j] - w1[:,k]
    # drop first term u0 from the weight matrix and add it in as a bias
    w1 = w1[:,1:]
    b1 = np.array(bool_funs)[:,0]
    n_model.layers[2].set_weights([w1.T,b1])
    return n_model

