import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx


def text_one_hot(x):
    """
    Create a onehot vector of length 7 representing the small input.

    x : integer from 1 to 128.
    """
    match x:
        case 32:  return [1,0,0,0,0,0,0]
        case 122: return [0,1,0,0,0,0,0]
        case 101: return [0,0,1,0,0,0,0]
        case 114: return [0,0,0,1,0,0,0]
        case 111: return [0,0,0,0,1,0,0]
        case 110: return [0,0,0,0,0,1,0]
        case _:   return [0,0,0,0,0,0,1]

def space_pad(text):
    return text + ' '

def text_to_tensor(text):
    """
    Create a small onehot encoded tensor input from a text string.

    text : input string.
    """
    text_num = [ord(x) for x in list(text)]
    text_conv = list(map(text_one_hot, text_num))
    text_tensor = tf.convert_to_tensor(text_conv, dtype=tf.float32)
    return tf.reshape(text_tensor, [1, text_tensor.shape[0], 7])

def text_to_onehot(text):
    """
    Create a full onehot encoded tensor input from a text string.

    text : input string.
    """
    text_num = [ord(x) for x in list(text)]
    text_array = np.zeros((len(text_num),128))
    for i,x in enumerate(text_num):
        text_array[i,x] = 1
    text_tensor = tf.convert_to_tensor(text_array, dtype=tf.float32)
    return tf.reshape(text_tensor, [1, text_tensor.shape[0], 128])


def make_rnn_model():
    """
    Create a model that detects "zero" and "one" within a small one-hot
    encoded tensor.
    
    The output is a sequence of vectors of length 2 representing a one-hot
    encoding of a detected word, or 0 if none are detected.
    """
    inputs = tf.keras.Input(shape=(None,7))
    recurrent1 = tf.keras.layers.SimpleRNN(10, return_sequences=True, activation='relu')(inputs)
    dense1 = tf.keras.layers.Dense(2, activation='linear')(recurrent1)
    rnn_model = tf.keras.Model(inputs=inputs, outputs=dense1, name='rnn_model')
    rnn_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    # recurrent1
    w0 = np.array([[ 0, 1, 0, 0, 0, 0, 0],
                   [ 0, 0, 1, 0, 0, 0, 0],
                   [ 0, 0, 0, 1, 0, 0, 0],
                   [ 0, 0, 0, 0, 1, 0, 0],
                   [ 0, 0, 0, 0, 1, 0, 0],
                   [ 0, 0, 0, 0, 0, 1, 0],
                   [ 0, 0, 1, 0, 0, 0, 0],
                   [ 1, 0, 0, 0, 0, 0, 0],
                   [ 1, 0, 0, 0, 0, 0, 0],
                   [-1, 1, 1, 1, 1, 1, 1]])

    w1 = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                   [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [ 0, 0,-1, 0, 0, 0, 0, 0, 0,-1],
                   [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    b0 = np.array([0,-1,-1,-1,0,-1,-1,-1,-1,0])

    rnn_model.layers[1].set_weights([w0.T, w1.T, b0.T])

    # dense1
    w2 = np.array([[0,0,0,0,0,0,0,1,0,0],
                   [0,0,0,0,0,0,0,0,1,0]])

    b1 = np.array([0,0])

    rnn_model.layers[2].set_weights([w2.T, b1.T])

    return rnn_model


def make_full_rnn_model():
    """
    Create a model that detects "zero" and "one" within a full one-hot
    encoded tensor.
    
    The output is a sequence of vectors of length 2 representing a one-hot
    encoding of a detected word, or 0 if none are detected.
    """
    inputs = tf.keras.Input(shape=(None,128))
    recurrent1 = tf.keras.layers.SimpleRNN(10, return_sequences=True, activation='relu')(inputs)
    dense1 = tf.keras.layers.Dense(2, activation='linear')(recurrent1)
    rnn_model = tf.keras.Model(inputs=inputs, outputs=dense1, name='full_rnn_model')
    rnn_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    # recurrent1
    w0 = np.zeros((10,128))
    for i in range(128):
        if i in [90, 122]:
            w0[:,i] = np.array([1,0,0,0,0,0,0,0,0,1])
        elif i in [69, 101]:
            w0[:,i] = np.array([0,1,0,0,0,0,1,0,0,1])
        elif i in [82, 114]:
            w0[:,i] = np.array([0,0,1,0,0,0,0,0,0,1])
        elif i in [79, 111]:
            w0[:,i] = np.array([0,0,0,1,1,0,0,0,0,1])
        elif i in [78, 110]:
            w0[:,i] = np.array([0,0,0,0,0,1,0,0,0,1])
        elif i in range(48,57) or i in range(65, 90) or i in range(97,122):
            w0[:,i] = np.array([0,0,0,0,0,0,0,0,0,1])
        else:
            w0[:,i] = np.array([0,0,0,0,0,0,0,1,1,-1])

    w1 = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                   [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [ 0, 0,-1, 0, 0, 0, 0, 0, 0,-1],
                   [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    b0 = np.array([0,-1,-1,-1,0,-1,-1,-1,-1,0])

    rnn_model.layers[1].set_weights([w0.T, w1.T, b0.T])

    # dense1
    w2 = np.array([[0,0,0,0,0,0,0,1,0,0],
                   [0,0,0,0,0,0,0,0,1,0]])

    b1 = np.array([0,0])

    rnn_model.layers[2].set_weights([w2.T, b1.T])

    return rnn_model
