import matplotlib.colors
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
from itertools import combinations


# As we are dealing with 6x6 images, set N=6 globally
N = 6
rng = np.random.default_rng()


# ---------- Functions for drawing digits ----------
def show_digit(A):
    plt.imshow(A, cmap='Greys')
    plt.axis("off")

def hl(A, i1, i2, j):
    """
    Horizontal line.
    """
    for p in range(i1, i2+1):
        A[j, p] = 1

def vl(A, i, j1, j2):
    """
    Vertical line.
    """
    for q in range(j1, j2+1):
        A[q, i] = 1

def ee(d):
    """
    One-hot representation of d.
    """
    u = np.zeros(10)
    u[d] = 1
    return u

"""
f{d} draws the digit d.
"""
def f0(i):
    if not(isinstance(i, list) and len(i) == 4):
        raise ValueError("i is not a list of length 4")
    if not(0 <= i[0] < i[1] - 1 < N - 1 and
           0 <= i[2] < i[3] - 1 < N - 1):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[2])
    hl(A, i[0], i[1], i[3])
    vl(A, i[0], i[2], i[3])
    vl(A, i[1], i[2], i[3])
    return A


def f1(i):
    if not(isinstance(i, list) and len(i) == 3):
        raise ValueError("i is not a list of length 3")
    if not(0 <= i[0] < N and
           0 <= i[1] < i[2] < N):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    vl(A, i[0], i[1], i[2])
    return A


def f2(i):
    if not(isinstance(i, list) and len(i) == 7):
        raise ValueError("i is not a list of length 7")
    if not(0 <= i[0] < i[1] < N and
           i[0] < i[2] < N and
           0 <= i[3] < i[2] and
           0 <= i[4] < i[5] - 1 < i[6] - 2 < N - 2):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[3], i[2], i[4])
    hl(A, i[0], i[2], i[5])
    hl(A, i[0], i[1], i[6])
    vl(A, i[2], i[4], i[5])
    vl(A, i[0], i[5], i[6])
    return A


def f3(i):
    if not(isinstance(i, list) and len(i) == 7):
        raise ValueError("i is not a list of length 7")
    if not(0 <= i[0] < i[1] < N and
           0 <= i[2] < i[1] and
           0 <= i[3] < i[1] and
           0 <= i[4] < i[5] - 1 < i[6] - 2 < N - 2):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[4])
    hl(A, i[2], i[1], i[5])
    hl(A, i[3], i[1], i[6])
    vl(A, i[1], i[4], i[6])
    return A


def f4(i):
    if not(isinstance(i, list) and len(i) == 6):
        raise ValueError("i is not a list of length 6")
    if not(0 <= i[0] < i[1] - 1 < N - 1 and
           0 <= i[2] < i[3] and
           0 <= i[4] < i[3] < i[5] < N ):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[3])
    vl(A, i[0], i[2], i[3])
    vl(A, i[1], i[4], i[5])
    return A


def f5(i):
    if not(isinstance(i, list) and len(i) == 7):
        raise ValueError("i is not a list of length 7")
    if not(0 <= i[0] < i[1] < N and
           0 <= i[2] < i[1] and
           i[2] < i[3] < N and
           0 <= i[4] < i[5] - 1 < i[6] - 2 < N - 2):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[6])
    hl(A, i[2], i[1], i[5])
    hl(A, i[2], i[3], i[4])
    vl(A, i[2], i[4], i[5])
    vl(A, i[1], i[5], i[6])
    return A


def f6(i):
    if not(isinstance(i, list) and len(i) == 5):
        raise ValueError("i is not a list of length 5")
    if not(0 <= i[0] < i[1] - 1 < N - 1 and
           0 <= i[2] < i[3] - 1 < i[4] - 2 < N - 2):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[4])
    hl(A, i[0], i[1], i[3])
    vl(A, i[0], i[2], i[4])
    vl(A, i[1], i[3], i[4])
    return A


def f7(i):
    if not(isinstance(i, list) and len(i) == 4):
        raise ValueError("i is not a list of length 4")
    if not(0 <= i[0] < i[1] < N and
           0 <= i[2] < i[3] < N):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[2])
    vl(A, i[1], i[2], i[3])
    return A


def f8(i):
    if not(isinstance(i, list) and len(i) == 5):
        raise ValueError("i is not a list of length 5")
    if not(0 <= i[0] < i[1] - 1 < N - 1 and
           0 <= i[2] < i[3] - 1 < i[4] - 2 < N - 2):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[2])
    hl(A, i[0], i[1], i[3])
    hl(A, i[0], i[1], i[4])
    vl(A, i[0], i[2], i[4])
    vl(A, i[1], i[2], i[4])
    return A


def f9(i):
    if not(isinstance(i, list) and len(i) == 5):
        raise ValueError("i is not a list of length 5")
    if not(0 <= i[0] < i[1] - 1 < N - 1 and
           0 <= i[2] < i[3] - 1 < i[4] - 2 < N - 2):
        raise ValueError("bad dimensions")
    A = np.zeros((N, N))
    hl(A, i[0], i[1], i[3])
    hl(A, i[0], i[1], i[2])
    vl(A, i[0], i[2], i[3])
    vl(A, i[1], i[2], i[4])
    return A


f = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]

"""
Create the possible images containing a digit.
"""
II = [
    [[i0, i1, i2, i3]
     for i0 in range(N-2)
     for i1 in range(i0+2, N)
     for i2 in range(N-2)
     for i3 in range(i2+2, N)],
    [[i0, i1, i2]
     for i0 in range(N)
     for i1 in range(N-1)
     for i2 in range(i1+1, N)],
    [[i0, i1, i2, i3, i4, i5, i6]
     for i0 in range(N-1)
     for i1 in range(i0+1, N)
     for i2 in range(i0+1, N)
     for i3 in range(i2)
     for i4 in range(N-4)
     for i5 in range(i4+2, N-2)
     for i6 in range(i5+2, N)],
    [[i0, i1, i2, i3, i4, i5, i6]
     for i0 in range(N-1)
     for i1 in range(i0+1, N)
     for i2 in range(i1)
     for i3 in range(i1)
     for i4 in range(N-4)
     for i5 in range(i4+2, N-2)
     for i6 in range(i5+2, N)],
    [[i0, i1, i2, i3, i4, i5]
     for i0 in range(N-2)
     for i1 in range(i0+2, N)
     for i2 in range(N-1)
     for i3 in range(i2+1, N)
     for i4 in range(i3)
     for i5 in range(i3+1, N)],
    [[i0, i1, i2, i3, i4, i5, i6]
     for i0 in range(N-1)
     for i1 in range(i0+1, N)
     for i2 in range(i1)
     for i3 in range(i2+1, N)
     for i4 in range(N-4)
     for i5 in range(i4+2, N-2)
     for i6 in range(i5+2, N)],
    [[i0, i1, i2, i3, i4]
     for i0 in range(N-2)
     for i1 in range(i0+2, N)
     for i2 in range(N-4)
     for i3 in range(i2+2, N)
     for i4 in range(i3+2, N)],
    [[i0, i1, i2, i3]
     for i0 in range(N-1)
     for i1 in range(i0+1, N)
     for i2 in range(N-1)
     for i3 in range(i2+1, N)],
    [[i0, i1, i2, i3, i4]
     for i0 in range(N-2)
     for i1 in range(i0+2, N)
     for i2 in range(N-4)
     for i3 in range(i2+2, N)
     for i4 in range(i3+2, N)],
    [[i0, i1, i2, i3, i4]
     for i0 in range(N-2)
     for i1 in range(i0+2, N)
     for i2 in range(N-4)
     for i3 in range(i2+2, N)
     for i4 in range(i3+2, N)]
]

IM = [[f[d](i) for i in II[d]] for d in range(10)]
x_all = np.array([img for d in range(10) for img in IM[d]]).astype("float32")
y_all = np.array([[[ee(d)]] for d in range(10) for img in IM[d]]).astype("float32")


def random_image(d=None):
    """
    Create a random image. The digit d can be specified.
    """
    if d is None:
        d = rng.integers(10)
    i = rng.integers(len(IM[d]))
    return IM[d][i]


# ---------------------------------------------------

def make_exact_model():
    """
    Create an exact model that can recognize all digits in IM and reject non-digits.
    The first layer consists of kernels to recognize each pixel type from 0 to 9.
    The second layer counts the number of occurrences of each pixel type in the
    first 10 channels, and then uses 7 channels to create ReLU parts that indicate
    the relative locations of the corner pixel types if they are present. The output
    layer determines the digit from this information.
    """
    inputs = tf.keras.Input(shape=(N, N))
    reshape = tf.keras.layers.Reshape((N, N, 1))(inputs)
    features = tf.keras.layers.Conv2D(10, [3, 3], [1, 1], 'same', activation='relu')(reshape)
    counts = tf.keras.layers.Conv2D(17, [N, N], [1, 1], 'valid', activation='relu')(features)
    outputs = tf.keras.layers.Dense(10, activation='relu')(counts)
    exact_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="exact_model")

    # features
    weights2 = [
            [[-1, -1,  0], [-1,  1,  1], [-1, -1,  0]],
            [[ 0, -1, -1], [ 1,  1, -1], [ 0, -1, -1]],
            [[-1, -1, -1], [-1,  1, -1], [ 0,  1,  0]],
            [[ 0,  1,  0], [-1,  1, -1], [-1, -1, -1]],
            [[-1, -1,  0], [-1,  1,  1], [ 0,  1, -1]],
            [[ 0, -1, -1], [ 1,  1, -1], [-1,  1,  0]],
            [[ 0,  1, -1], [-1,  1,  1], [-1, -1,  0]],
            [[-1,  1,  0], [ 1,  1, -1], [ 0, -1, -1]],
            [[-1,  1, -1], [-1,  1,  1], [-1,  1, -1]],
            [[-1,  1, -1], [ 1,  1, -1], [-1,  1, -1]]
    ]
    bias2 = [-1, -1, -1, -1, -2, -2, -2, -2, -3, -3]
    weights2 = np.array(weights2).transpose((1, 2, 0)).reshape((3, 3, 1, 10))
    bias2 = np.array(bias2)
    exact_model.layers[2].set_weights([weights2, bias2])

    # counts
    weights3 = np.zeros((N, N, 10, 17))
    for j in range(N):
        for k in range(N):
            for p in range(10):
                weights3[j, k, p, p] = 1

            weights3[j, k, 4, 10] =  j
            weights3[j, k, 5, 10] = -j
            weights3[j, k, 6, 10] =  j
            weights3[j, k, 7, 10] = -j
            weights3[j, k, 4, 11] =  j
            weights3[j, k, 5, 11] = -j
            weights3[j, k, 6, 11] =  j
            weights3[j, k, 7, 11] = -j

            weights3[j, k, 4, 12] = -j
            weights3[j, k, 5, 12] =  j
            weights3[j, k, 6, 12] = -j
            weights3[j, k, 7, 12] =  j
            weights3[j, k, 4, 13] = -j
            weights3[j, k, 5, 13] =  j
            weights3[j, k, 6, 13] = -j
            weights3[j, k, 7, 13] =  j

            weights3[j, k, 4, 14] =  k
            weights3[j, k, 6, 14] = -k
            weights3[j, k, 4, 15] =  k
            weights3[j, k, 6, 15] = -k
            weights3[j, k, 4, 16] =  k
            weights3[j, k, 6, 16] = -k
    bias3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 1, 0, -1])
    exact_model.layers[3].set_weights([weights3, bias3])

    # output
    weights4 = np.array([
        [-1, -1, -1, -1,  1,  1,  1,  1, -1, -1,  0,  0,  0,  0,  0,  0,  0],
        [-1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0],
        [ 1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  0,  0,  1, -2,  1],
        [ 1, -1, -1, -1, -1,  1, -1,  1, -1,  1,  0,  0,  0,  0,  0,  0,  0],
        [-1, -1,  1,  1, -1, -1,  1, -1, -1,  1,  0,  0,  0,  0,  0,  0,  0],
        [ 1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  0,  0,  1, -1,  1, -2,  1],
        [-1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  0,  0,  0,  0,  0,  0,  0],
        [ 1, -1, -1,  1, -1,  1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0],
        [-1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0],
        [-1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  0,  0,  0,  0,  0,  0,  0]
    ]).transpose()
    bias4 = np.array([-3, -1, -7, -5, -4, -6, -4, -2, -5, -4])
    exact_model.layers[4].set_weights([weights4, bias4])

    return exact_model


def model_classify(M, img):
    """
    Returns the classification of an image using an existing model M.

    M : model
    img : input image
    """
    p = M.predict(img.reshape(1, N, N), verbose=0)
    if np.sum(p) == 0: return "Reject"
    else: return tf.math.argmax(p[0][0][0]).numpy()


def make_approx_model(p=3, q=3, r=0.001):
    """
    Creates an approximate model with the same layers as the exact model but p
    channels in the first layer and q channels in the second layer. r is the
    learning rate which is assigned at model compilation.

    p : positive integer
    q : positive integer
    r : float
    """
    inputs = tf.keras.Input(shape=(N, N))
    reshape = tf.keras.layers.Reshape((N, N, 1))(inputs)
    features = tf.keras.layers.Conv2D(p, [3, 3], [1, 1], 'same', activation='relu')(reshape)
    counts = tf.keras.layers.Conv2D(q, [N, N], [1, 1], 'valid', activation='relu')(features)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(counts)
    approx_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="approx_model")
    opt = tf.keras.optimizers.Adam(learning_rate=r)
    approx_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return approx_model


def train_approx_model(M, epochs, batch_size=32, patience=20):
    """
    Trains an approximate model M using the x_all and y_all global objects. The training
    parameters other than the learning rate can be adjusted.

    M : approximate model
    epochs : positive integer
    batch_size : positive integer
    patience : positive integer
    """
    all_dataset = tf.data.Dataset.from_tensor_slices((x_all, y_all))
    all_dataset = all_dataset.shuffle(100).batch(batch_size)
    if isinstance(patience, int):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, verbose=0)
        return M.fit(all_dataset, epochs=epochs, callbacks=[callback], verbose=0)
    else: return M.fit(all_dataset, epochs=epochs, verbose=0)


def check_model(M):
    """
    Checks how many of the possible input images containing a digit the
    model M can classify correctly.

    M : model
    """
    z_all = np.argmax(y_all.reshape((-1, 10)), axis=1)
    z_all_pred = np.argmax(M(x_all).numpy().reshape((-1, 10)), axis=1)
    return sum(z_all == z_all_pred)


def show_kernels(M):
    """
    Shows the kernels of the first layer for a model M. Uses RGB if there are
    3 channels.

    M : model
    """
    l = M.layers
    p = l[3].input_shape[3]
    w = M.layers[2].get_weights()[0].reshape(3, 3, p).transpose(2, 0, 1)
    fig = plt.figure(figsize=(2*p-1, 1))
    for i in range(p):
        if p == 3:
            cm0 = np.zeros([11, 4])
            for j in range(11):
                cm0[j, i] = j / 10
                cm0[j, 3] = 1
            cm = matplotlib.colors.ListedColormap(cm0)
        else:
            cm = 'Greys'
        x = (2*i - 0.02) / (2*p-1)
        ax = fig.add_axes([x, 0.02, 0.96 / (2*p-1), 0.96])
        ax.set_axis_off()
        ax.imshow(w[i], cmap=cm)


def make_approx_model_b(p=3, q=3, r=0.001):
    """
    An alternative to make_approx_model() that was not used but verifies that the behaviour
    is consistent when flattening the feature map and using a dense layer.

    p : positive integer
    q : positive integer
    r : float
    """
    inputs = tf.keras.Input(shape=(N, N))
    reshape = tf.keras.layers.Reshape((N, N, 1))(inputs)
    features = tf.keras.layers.Conv2D(p, [3, 3], [1, 1], 'same', activation='relu')(reshape)
    flatten = tf.keras.layers.Flatten()(features)
    counts = tf.keras.layers.Dense(q, activation='relu')(flatten)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(counts)
    reshape_out = tf.keras.layers.Reshape((1,1,10))(outputs)
    approx_model = tf.keras.Model(inputs=inputs, outputs=reshape_out, name="approx_model")
    opt = tf.keras.optimizers.Adam(learning_rate=r)
    approx_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return approx_model


def test_starting_kernel(sk, r, epochs, batch_size, patience):
    """
    Creates and approximate model and trains it with the first layer kernels
    initialized to be the kernels for pixel types in the list sk. Returns the
    final model and the training history.

    sk : list of 3 integers from 0 to 9
    r : float
    epochs : positive integer
    batch_size : positive integer
    patience : positive integer
    """
    M_exact = make_exact_model()
    w_exact = M_exact.layers[2].get_weights()[0][:,:,:,sk]
    b_exact = M_exact.layers[2].get_weights()[1][sk]

    M = make_approx_model(p=3, q=3, r=r)
    M.layers[2].set_weights([w_exact,b_exact])

    hist = train_approx_model(M=M, epochs=epochs, batch_size=batch_size, patience=patience)

    return M, hist


def test_all():
    """
    Runs test_starting_kernel() for starting kernels of the form [d,d,d] for
    d in 0 to 9. Uses the fixed training parameters which we found to be
    suitable. Repeats 5 tests for each starting kernel and returns the mean
    scores from check_model() and the full list of trained models.
    """
    scores = []
    models = []
    for i in range(10):
        score = []
        model = []
        for j in range(5):
            M, hist = test_starting_kernel(sk=[i,i,i], r=0.003, epochs=1000, batch_size=128, patience=40)
            score.append(check_model(M=M))
            model.append(M)
        scores.append(np.mean(np.array(score)))
        models.append(model)
    return scores, models
