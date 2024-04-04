import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os


def text_one_hot(x):
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
    text_num = [ord(x) for x in list(text)]
    text_conv = list(map(text_one_hot, text_num))
    text_tensor = tf.convert_to_tensor(text_conv, dtype=tf.float32)
    return tf.reshape(text_tensor, [1, text_tensor.shape[0], 7])


def make_rnn_model():
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

"""
M = make_rnn_model()
text = 'zero one '
textt = text_to_tensor(text)
M.predict(textt)
"""


"""
states for the next model will be:
nothing, z, ze, zer, zero, zero (out), o, on, one, one (out)

state satrts initialised as [1,0,0,0,0,0,0,0,0,0]

basic structure of combining current state with input remains the same

with the neutral state as a 1, it is easier to show which states it can move to

"""

def make_init_rnn_model():
    h0 = np.zeros((1,10))
    h0[0,0] = 1
    kh0 = tf.keras.backend.constant(h0)

    inputs = tf.keras.Input(shape=(None,7))
    recurrent1 = tf.keras.layers.SimpleRNN(10, return_sequences=True, activation='relu')(inputs, initial_state=kh0)
    dense1 = tf.keras.layers.Dense(2, activation='linear')(recurrent1)
    rnn_model = tf.keras.Model(inputs=inputs, outputs=recurrent1, name='rnn_model')
    rnn_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    # recurrent1
    w0 = np.array([[ 1, 1, 1, 1, 1, 1, 1], # any character can reset the state
                   [ 0, 1, 0, 0, 0, 0, 0], # z can move to z
                   [ 0, 0, 1, 0, 0, 0, 0], # e can move to ze
                   [ 0, 0, 0, 1, 0, 0, 0], # r can move to zer
                   [ 0, 0, 0, 0, 1, 0, 0], # o can move to zero
                   [ 0, 0, 0, 0, 1, 0, 0], # o can move to o
                   [ 0, 0, 0, 0, 0, 1, 0], # n can move to on
                   [ 0, 0, 1, 0, 0, 0, 0], # e can move to one
                   [ 1, 0, 0, 0, 0, 0, 0], # space can move to zero (out)
                   [ 1, 0, 0, 0, 0, 0, 0]]) # space can move to one (out)
                  # init z ze zer zero o on one zero one
    w1 = np.array([[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # init can move to z or o
                   [ 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], # z can move to ze or init
                   [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # ze can move to zer or init
                   [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # zer can move to zero or init
                   [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # zero can move to zero (out) or init
                   [ 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], # o can move to on or init
                   [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # on can move to one or init
                   [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # one can move to one (out) or init
                   [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # zero (out) can move to z, o, or init
                   [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]) # zero (one) can move to z, o, or init

    b0 = np.array([0,0,0,0,0,0,0,0,0,0])
    b0 = -np.ones(10)

    rnn_model.layers[1].set_weights([w0.T, w1.T, b0.T])

    # dense1
    w2 = np.array([[0,0,0,0,0,0,0,1,0,0],
                   [0,0,0,0,0,0,0,0,1,0]])

    b1 = np.array([0,0])

    #rnn_model.layers[2].set_weights([w2.T, b1.T])

    return rnn_model

"""
M = make_rnn_model()
text = 'zero one '
textt = text_to_tensor(text)
M.predict(textt)
"""