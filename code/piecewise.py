import tensorflow as tf
import numpy as np
import itertools
import math
from matplotlib import pyplot as plt
import os


# f(x) = c + m0*x + sum((mi - m_i-1)(x - ai))
# need to add x = (x)_+ - (-x)_+ to first layer!

# make model for the R -> R case
def make_oned_model(a, m, c):
    """
    Create a model that evaluates a piecewise linear function specified in one dimension.

    The first layer creates r+1 variables using the ReLU activation layer, where the
    first 2 variables are (x)_+ and (-x)_+ to ensure x can be retrieved in the second
    layer. The other variables are (x-a)_+
    The second layer combines them according to weights derived from the gradients
    supplied in m.
    Predictions can be made on a real number x, supplied as a tensor.

    a : np.array with r entries.
    m : np.array with r+1 entries.
    c : any float.
    """
    r = int(m.shape[0])
    inputs = tf.keras.Input(shape=(1))
    dense1 = tf.keras.layers.Dense(r+1, activation='relu')(inputs)
    dense2 = tf.keras.layers.Dense(1, activation='linear')(dense1)
    oned_model = tf.keras.Model(inputs=inputs, outputs=dense2, name='oned_model')
    oned_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    w0 = np.ones((1,r+1))
    w0[0,1] = -1
    b0 = np.concatenate(([0]*2,-a))
    oned_model.layers[1].set_weights([w0,b0])

    w1 = np.concatenate((np.array([m[0]]),np.diff(m, prepend=0)))
    w1[1] = -w1[1]
    w1 = w1.reshape(r+1,1)
    b1 = np.array([c])
    oned_model.layers[2].set_weights([w1,b1])
    return oned_model


a0 = np.array([0,1,2,3])
m0 = np.array([1,2,4,6,8])
c0 = 1

make_exact_model(a0, m0, c0)
exact_model.predict(np.array([0]).reshape(1,1),verbose=0)

# 2D case

# what if we have a_1 to a_r and b_1 to b_s
# and m_0 to m_r and l_0 to l_s

# we then construct plane gradients
# m00 ... m0r m11 ... msr
# where mij = [mi, ls]

# this construction ensures that
# 1) each grid in (x,y) has a constant gradient i.e. f is linear
# 2) for fixed i, the x gradient mi is constant and so at the boundaries
#    between x<= ai and x>ai there are no discontinuities

# we can now moadify the above 1D case so that a and m are 2D arrays
# containing the points and partial derivatives in each dimension



def make_nd_model(a, m, c):
    """
    Create a model that evaluates a piecewise linear function specified in d >= 1
    dimensions.

    The first layer creates r+d tuples using the ReLU activation layer. This
    replicates the one-dimensional case over d columns, where r is the sum of r_i.
    The second layer combines them according to weights derived from the gradients
    supplied in m.
    Predictions can be made on a tensor with d real entries.

    a : list of np.arrays, where each entry has its own length r_i.
    m : list of np.arrays, where each entry has its own length r_i + 1.
    c : any float.
    """
    d = len(m)
    r = sum([np.shape(m_i)[0] for m_i in m])
    inputs = tf.keras.Input(shape=(d))
    dense1 = tf.keras.layers.Dense(r, activation='relu')(inputs)
    dense2 = tf.keras.layers.Dense(1, activation='linear')(dense1)
    nd_model = tf.keras.Model(inputs=inputs, outputs=dense2, name='nd_model')
    nd_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    # for multiple inputs, use block "diagonal" weights in first layer
    # to handle each input separately
    w0 = np.zeros((d,r))
    x=0
    y=0
    for i in range(d):
        y = y + np.shape(m[i])[0]
        w0[i,x:y] = np.ones(y-x)
        x = x + y
    b0 = -np.concatenate([np.concatenate((np.array([0]),a_i)) for a_i in a0])
    nd_model.layers[1].set_weights([w0,b0])

    # output of the first dense layer is a list so can just concatenate
    # the diffs from before
    w1 = np.concatenate([np.diff(m_i, prepend=0) for m_i in m]).reshape(r,1)
    b1 = np.array([c])
    nd_model.layers[2].set_weights([w1,b1])
    return nd_model

    


a0 = [np.array([0,1,2,3]), np.array([0,1,2])]
m0 = [np.array([1,4,9,16,25]), np.array([1,4,9,16])]
c0 = 1


make_exact_nd_model(a0, m0, c0)
exact_nd_model.predict(np.array([2,2]).reshape(1,2),verbose=0)