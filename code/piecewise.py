import tensorflow as tf
import numpy as np
import itertools
import math
from matplotlib import pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D


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
    dense1 = tf.keras.layers.Dense(r+d, activation='relu')(inputs)
    dense2 = tf.keras.layers.Dense(1, activation='linear')(dense1)
    nd_model = tf.keras.Model(inputs=inputs, outputs=dense2, name='nd_model')
    nd_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

    # for multiple inputs, use block "diagonal" weights in first layer
    # to handle each input separately
    w0 = np.zeros((d,r+d))
    b0 = np.zeros(r+d)
    x=0
    y=0
    for i in range(d):
        y = y + np.shape(m[i])[0] + 1
        w0[i,x:y] = np.ones(y-x)
        w0[i,x+1] = -1
        b0[x:y] = -np.concatenate((np.array([0]*2), a[i]))
        x = x + y
    nd_model.layers[1].set_weights([w0,b0])

    # concatenate the weights for each dimension
    w1 = np.zeros(r+d)
    x=0
    y=0
    for i in range(d):
        y = y + np.shape(m[i])[0] + 1
        w1[x:y] = np.concatenate((np.array([m[i][0]]), np.diff(m[i], prepend=0)))
        w1[x+1] = -w1[x+1]
        x = x + y
    w1 = w1.reshape(r+d,1)
    b1 = np.array([c])
    nd_model.layers[2].set_weights([w1,b1])
    return nd_model


def eval_oned_np(x, M):
    """
    Create a wrapper for a one dimensional model M so that its input and output are floats.

    x : input float.
    M : model created by make_oned_model.
    """

    xt = tf.convert_to_tensor(np.array(x).reshape(1,1))
    yt = M.predict(xt, verbose=0)
    return float(yt)


def plot_oned_model(M, xlim=(-10,10), pt=1000, title=None):
    """
    Create a plot from the one dimensional model M.

    M : model created by make_oned_model.
    xlim : the limits between which to plot the graph y=M(x).
    pt : number of points to evaluate on within the range.
    """
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    x = np.linspace(xlim[0], xlim[1], pt)
    xt = tf.convert_to_tensor(x.reshape(pt,1))
    y = M.predict(xt, verbose=0)
    plt.plot(x, y)

    if title != None: plt.title(title)
    plt.xlabel('x')
    plt.ylabel('f(x)')

    plt.show()


def plot_twod_model(M, x1lim=(-10,10), x2lim=(-10,10), pt=100, title=None):
    """
    Create a plot from the two dimensional model M.

    M : two dimensional model created by make_nd_model.
    x1lim : the limits between which to plot the graph y=M(x).
    x2lim : the limits between which to plot the graph y=M(x).
    pt : number of points to evaluate on within the range.
    """

    x1 = np.linspace(x1lim[0], x1lim[1], pt)
    x2 = np.linspace(x2lim[0], x2lim[1], pt)
    X1, X2 = np.meshgrid(x1, x2)
    x = np.dstack((X1,X2)).reshape(-1,2)

    xt = tf.convert_to_tensor(x)
    y = M.predict(xt, verbose=0)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X1, X2, y.reshape(pt,pt))

    if title != None: plt.title(title)
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    plt.show()


def make_approx_model(N, r=0.01):
    """
    Initialize a model that can be trained to approximate an exact one dimensional model.
    It will use the same architecture as the given model N.

    N : exact model created by make_oned_model.
    r : the learning rate for Adam.
    """

    M = tf.keras.models.clone_model(N)
    opt = tf.keras.optimizers.Adam(learning_rate=r)
    M._name = 'approx_model'
    M.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error"])
    return M


def train_approx_model(M, N, xlim=(-10,10), samples=1000, epochs=100):
    """
    Train the model M using samples generated from the model N.

    M : approx model initialized using make_approx_model.
    N : exact model created by make_oned_model.
    """

    x = np.linspace(xlim[0], xlim[1], samples)
    xt = tf.convert_to_tensor(x.reshape(samples,1))
    y = N.predict(xt, verbose=0)
    data = tf.data.Dataset.from_tensors((xt, y))
    M.fit(data, epochs=epochs, verbose=0)
    return M


def plot_trained_model(M, N, xlim=(-10,10), pt=1000, title=None):
    """
    Create a plot to compare a trained one dimensional model M with an
    exact one dimensional model N.

    M : approx model.
    N : exact model created by make_oned_model.
    xlim : the limits between which to plot the graph y=M(x).
    pt : number of points to evaluate on within the range.
    """
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    x = np.linspace(xlim[0], xlim[1], pt)
    xt = tf.convert_to_tensor(x.reshape(pt,1))

    yM = M.predict(xt, verbose=0)
    yN = N.predict(xt, verbose=0)

    plt.plot(x, yN, label='Exact model')
    plt.plot(x, yM, label='Trained model', color='red')

    if title != None: plt.title(title)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    plt.show()