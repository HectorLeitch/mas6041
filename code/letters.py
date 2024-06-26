# Try example 6.5
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os


letters = "iyjcolhtux"

letter_pixels = {
    "i": np.array([[0,1,0],[0,1,0],[0,1,0]]),
    "y": np.array([[1,0,1],[0,1,0],[0,1,0]]),
    "j": np.array([[0,0,1],[0,0,1],[1,1,1]]),
    "c": np.array([[1,1,1],[1,0,0],[1,1,1]]),
    "o": np.array([[1,1,1],[1,0,1],[1,1,1]]),
    "l": np.array([[1,0,0],[1,0,0],[1,1,1]]),
    "h": np.array([[1,0,1],[1,1,1],[1,0,1]]) ,
    "t": np.array([[1,1,1],[0,1,0],[0,1,0]]),
    "u": np.array([[1,0,1],[1,0,1],[1,1,1]]),
    "x": np.array([[1,0,1],[0,1,0],[1,0,1]])
}

def ascii_pixel(c):
    return ('#' if c else ' ')

def ascii_image(l):
    return '\n'.join(map(lambda r: str.join('',map(ascii_pixel,r)),l))


def write_letter(grid, letter, x, y):
    """
    Write a letter into a grid at position (x,y)

    The grid is expected to be a numpy array of shape (h,w) say.
    The letter is expected to be a string from the string letters
    defined above.  The position (x,y) is expected to be a pair of 
    integers with 0 <= x < w-2 and 0 <= y < h-2.  Note that the 
    horizontal coordinate is given first, and the vertical coordinate 
    counts down from the top of the grid.
    """
    h, w = np.shape(grid)
    if y > h - 3 or x > w - 3:
        print(f"In grid of width {w} and height {h}, letter cannot be added with  (x,y)=({x},{y})")
        return grid
    elif not letter in letters:
        print(f"Letter {letter} not recognised")
        return grid
    else:
        grid[y:y+3,x:x+3] = letter_pixels[letter]
        return grid

def make_ragged_grid(word, width=None, height=6):
    """
    Make a grid with a word written in it.
    
    The word is expected to be a string from the string letters defined above.
    If the width is not specified, it is taken to be 5 times the length of the word.
    The letters are placed in the grid in a roughly uniform way, but with some
    perturbation both horizontally and vertically.
    """
    n = len(word)
    if width is None:
        width = 5*n
    grid = np.zeros([height,width])
    if n == 0:
        return grid
    for i, c in enumerate(word):
        x_pos = np.random.randint(np.round((i*width)/n), np.round(((i+1)*width)/n-3))
        y_pos = np.random.randint(0,height-3)
        grid = write_letter(grid,word[i],x_pos,y_pos)
    return grid


def make_explicit_model():
    """
    Create a model that recognises a letter given in a 3x3 grid.
    
    The input is passed to a convolutional layer with kernel size 3x3, effectively
    performing elementwise multiplication and taking the sum before applying
    the activation and bias.
    """
    inputs = tf.keras.Input(shape=(3, 3))
    reshape = tf.keras.layers.Reshape((3, 3, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(10, [3, 3], [1, 1], padding='valid', activation='relu')(reshape)
    explicit_model = tf.keras.Model(inputs=inputs, outputs=convlayer, name="explicit_model")
    explicit_model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    w0 = np.array([[-1,  1, -1,  1,  1,  1,  1,  1,  1,  1],
                   [ 1, -1, -1,  1,  1, -1, -1,  1, -1, -1],
                   [-1,  1,  1,  1,  1, -1,  1,  1,  1,  1],
                   [-1, -1, -1,  1,  1,  1,  1, -1,  1, -1],
                   [ 1,  1, -1, -1, -1, -1,  1,  1, -1,  1],
                   [-1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1],
                   [ 1,  1,  1,  1,  1,  1, -1,  1,  1, -1],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1]])
    w1 = w0.reshape(3,3,1,10)
    b0 = np.array([-2, -3, -4, -6, -7, -4, -6, -4, -6, -4])
    explicit_model.layers[2].set_weights([w1, b0])

    return explicit_model



def test_explicit_model(M):
    """
    Tests the explicit model above for all possible 2^9=512 3x3 images.

    M : model created by make_explicit_model().
    """
    all_images = np.array(list(itertools.product(*[[0,1]]*9))).reshape(512,1,3,3)
    y_all = np.zeros((512,1,1,10))
    for i in range(10):
        for j in range(512):
            if np.array_equal(all_images[j,0],letter_pixels[letters[i]]):
                y_all[j,0,0,i] = 1
                break
    return np.array_equal(M.predict(all_images, verbose=0), y_all)


# 11 channels, 1 per letter (inluding reject)
# 3x3 filter shape
# Strides of 1 horizontally, 1 vertically

# For max pooling use the height of the input-1, 3-wide, stride 3 across
# to find each letter detected once
# Stride height of input-1 limits the output to 1 row
def make_conv_model(h,w):
    """
    Create a model that applies the convolutional layer over a grid of height h
    and width w.
    
    The output is a sequence of vectors of length 10 representing a one-hot
    encoding of a detected letter, or 0 if none are detected.

    h : integer, height of input image
    w : integer, width of input image
    """
    inputs = tf.keras.Input(shape=(h, w))
    reshape = tf.keras.layers.Reshape((h, w, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(10, [3, 3], [1, 1], padding='valid', activation='relu')(reshape)
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(5,3), strides=(5,3), padding='same')(convlayer)
    conv_model = tf.keras.Model(inputs=inputs, outputs=pooling, name="conv_model")
    conv_model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    w0 = np.array([[-1,  1, -1,  1,  1,  1,  1,  1,  1,  1,],
                   [ 1, -1, -1,  1,  1, -1, -1,  1, -1, -1,],
                   [-1,  1,  1,  1,  1, -1,  1,  1,  1,  1,],
                   [-1, -1, -1,  1,  1,  1,  1, -1,  1, -1,],
                   [ 1,  1, -1, -1, -1, -1,  1,  1, -1,  1,],
                   [-1, -1,  1, -1,  1, -1,  1, -1,  1, -1,],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1,],
                   [ 1,  1,  1,  1,  1,  1, -1,  1,  1, -1,],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1,]])
    w1 = w0.reshape(3,3,1,10)
    b0 = np.array([-2, -3, -4, -6, -7, -4, -6, -4, -6, -4])
    conv_model.layers[2].set_weights([w1, b0])

    return conv_model


# CONSIDER CHANGING THIS TO SUM REJECTS WITH A GRU

# The example notes suggest two extensions
# The first is keeping a running total of the counts of each letter
# Add relu(2x) to the conv layer from above which results in one-hot encoding for the letters
# Pass into recurrent layer with 0 bias and identity weights to simply add each input
# into the memory
def make_reject_model(h,w):
    inputs = tf.keras.Input(shape=(h, w))
    reshape1 = tf.keras.layers.Reshape((h, w, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(10, [3, 3], [1, 1], padding='valid', activation='relu')(reshape1)
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(h,3), strides=(h,3), padding='same')(convlayer)
    reshape2 = tf.keras.layers.Reshape((h, 10))(pooling)
    recurrent = tf.keras.layers.SimpleRNN(10, return_sequences=False, activation='relu')(reshape2)
    sum_model = tf.keras.Model(inputs=inputs, outputs=recurrent, name="sum_model")
    sum_model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    # Convolution layer - as above
    w0 = np.array([[-1,  1, -1,  1,  1,  1,  1,  1,  1,  1],
                   [ 1, -1, -1,  1,  1, -1, -1,  1, -1, -1],
                   [-1,  1,  1,  1,  1, -1,  1,  1,  1,  1],
                   [-1, -1, -1,  1,  1,  1,  1, -1,  1, -1],
                   [ 1,  1, -1, -1, -1, -1,  1,  1, -1,  1],
                   [-1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1],
                   [ 1,  1,  1,  1,  1,  1, -1,  1,  1, -1],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1]])   
    w1 = w0.reshape(3,3,1,10)
    b0 = np.array([-2, -3, -4, -6, -7, -4, -6, -4, -6, -4])
    sum_model.layers[2].set_weights([w1, b0]) 

    w2 = np.eye(10)
    b1 = np.zeros(10)
    sum_model.layers[5].set_weights([w2, w2, b1])

    return sum_model

"""

G = make_ragged_grid('ccty')
SM = make_sum_model(G.shape[0], G.shape[1])
SM.predict(G.reshape(1,6,20))


"""

def make_sum_model(h, w):
    """
    Create a model that applies the convolutional layer over a grid of height h
    and width w, and adds the counts for each letter detected.
    
    The output is a single vector of length 10 with the counts for each letter.

    h : integer, height of input image
    w : integer, width of input image
    """
    w_g = int(np.floor(w/3))
    inputs = tf.keras.Input(shape=(h, w))
    reshape1 = tf.keras.layers.Reshape((h, w, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(10, [3, 3], [1, 1], padding='valid', activation='relu')(reshape1)
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(h,3), strides=(h,3), padding='same')(convlayer)
    reshape2 = tf.keras.layers.Permute((1,3,2))(pooling)
    sumlayer = tf.keras.layers.Dense(1, activation='linear')(reshape2)
    pq_conv_model = tf.keras.Model(inputs=inputs, outputs=sumlayer, name='pq_conv_model')
    pq_conv_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    w0 = np.array([[-1,  1, -1,  1,  1,  1,  1,  1,  1,  1],
                   [ 1, -1, -1,  1,  1, -1, -1,  1, -1, -1],
                   [-1,  1,  1,  1,  1, -1,  1,  1,  1,  1],
                   [-1, -1, -1,  1,  1,  1,  1, -1,  1, -1],
                   [ 1,  1, -1, -1, -1, -1,  1,  1, -1,  1],
                   [-1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1],
                   [ 1,  1,  1,  1,  1,  1, -1,  1,  1, -1],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1]])
    w1 = w0.reshape(3,3,1,10)
    b0 = np.array([-2, -3, -4, -6, -7, -4, -6, -4, -6, -4])
    pq_conv_model.layers[2].set_weights([w1, b0])

    w2 = np.ones([w_g,1])
    b1 = np.zeros(1)
    pq_conv_model.layers[5].set_weights([w2, b1])

    return pq_conv_model


def make_gru_model(h, w):
    """
    Create a model that recognises words within a grid of height h and width w.
    
    The output is a single vector of length floor(w/3) representing the detected
    word. It is assumed the grid contains a single word written from left to right.
    The steps are described in chapter 5 of the dissertation.

    h : integer, height of input image
    w : integer, width of input image
    """
    w_g = int(np.floor(w/3))
    inputs = tf.keras.Input(shape=(h, w))
    reshape1 = tf.keras.layers.Reshape((h, w, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(10, [3, 3], [1, 1], padding='valid', activation='relu')(reshape1)
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(h,3), strides=(h,3), padding='same')(convlayer)
    reshape2 = tf.keras.layers.Reshape((w_g,10))(pooling)

    # Dense1 creates (x_t, I(x_t > 0))
    dense1 = tf.keras.layers.Dense(2,activation='linear')(reshape2)

    # Recurrent1 creates s_t
    recurrent1 = tf.keras.layers.GRU(1, activation='linear', recurrent_activation='linear', return_sequences=True)(dense1)
    
    # Dense2 and dense3 produce the position indicators p_t
    dense2 = tf.keras.layers.Dense(3*w_g, activation='relu')(recurrent1)
    dense3 = tf.keras.layers.Dense(w_g, activation='linear')(dense2)

    # Merge dense layers and compute z using 2 layer bool
    merge1 = tf.keras.layers.Concatenate(axis=-1)([dense1, dense3])
    dense4 = tf.keras.layers.Dense(w_g, activation='relu')(merge1)

    # Recurrent2 returns the original sequence with the non-zeros first
    merge2 = tf.keras.layers.Concatenate(axis=-1)([dense1, dense4])
    recurrent2 = tf.keras.layers.GRU(w_g, activation='linear', recurrent_activation='relu', return_sequences=False, reset_after=False)(merge2)
    
    gru_model = tf.keras.Model(inputs=inputs, outputs=recurrent2, name='gru_model')
    gru_model.compile(loss='mean_squared_error', metrics=['accuracy'])

    # Weight assignments
    w0 = np.array([[-1,  1, -1,  1,  1,  1,  1,  1,  1,  1],
                   [ 1, -1, -1,  1,  1, -1, -1,  1, -1, -1],
                   [-1,  1,  1,  1,  1, -1,  1,  1,  1,  1],
                   [-1, -1, -1,  1,  1,  1,  1, -1,  1, -1],
                   [ 1,  1, -1, -1, -1, -1,  1,  1, -1,  1],
                   [-1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1],
                   [ 1,  1,  1,  1,  1,  1, -1,  1,  1, -1],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1]])
    w1 = w0.reshape(3,3,1,10)
    b0 = np.array([-2, -3, -4, -6, -7, -4, -6, -4, -6, -4])
    gru_model.layers[2].set_weights([w1, b0])

    # Dense1
    w2 = np.array([[1,1],[2,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1],[9,1],[10,1]])
    b1 = np.zeros(2)
    gru_model.layers[5].set_weights([w2,b1])

    # Recurrent1
    w3 = np.array([[0,0,0],[-1,0,0]]) #z_t = I(x_t =< 0) = 1 - I(x_t > 0)
    w4 = np.array([[0,0,1]]) #h^_t = h_t-1 + 1
    b2 = np.array([[1,1,1],[0,0,0]]) #r_t = 1, h^_t = h_t-1 + 1
    gru_model.layers[6].set_weights([w3, w4, b2])

    # Dense2
    w5 = np.ones((1,3*w_g))
    b3 = np.repeat(-np.arange(w_g),3)
    b3[1::3] = -np.arange(w_g)-1
    b3[2::3] = -np.arange(w_g)-2
    gru_model.layers[7].set_weights([w5, b3])

    # Dense3
    w6 = np.zeros((3*w_g, w_g))
    for j in range(w_g):
        w6[(j*3):((j+1)*3),j] = np.array([1,-2,1])
    b4 = np.zeros(w_g)
    gru_model.layers[8].set_weights([w6, b4])

    # Dense4
    w7 = np.zeros((2+w_g, w_g))
    w7[1,:] = np.ones(w_g)
    w7[2:(2+w_g),:] = np.eye(w_g)
    b5 = -np.ones(w_g)
    gru_model.layers[10].set_weights([w7, b5])

    # Recurrent2
    w8 = np.zeros((2+w_g, 3*w_g))    
    w8[0,(2*w_g):(3*w_g)] = np.ones(w_g) # r^t = 1, h^_t = x_t
    w8[2:(2+w_g),0:w_g] = -np.eye(w_g) # z_t = 1 - I(s=j and x>0)
    w9 = np.zeros((w_g, 3*w_g))
    b6 = np.zeros(3*w_g)
    b6[0:w_g] = np.ones(w_g) # z_t = 1 - I(s=j and x>0)
    gru_model.layers[12].set_weights([w8, w9, b6])

    return gru_model


def make_alt_gru_model(h, w):
    """
    Create a model that recognises words within a grid of height h and width w.
    The architecture is slightly modified as described in chapter 5.
    
    The output is a single vector of length floor(w/3) representing the detected
    word. It is assumed the grid contains a single word written from left to right.
    The steps are described in chapter 5 of the dissertation.

    h : integer, height of input image
    w : integer, width of input image
    """
    w_g = int(np.floor(w/3))
    inputs = tf.keras.Input(shape=(h, w))
    reshape1 = tf.keras.layers.Reshape((h, w, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(10, [3, 3], [1, 1], padding='valid', activation='relu')(reshape1)
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(h,3), strides=(h,3), padding='same')(convlayer)
    reshape2 = tf.keras.layers.Reshape((w_g, 10))(pooling)
    dense1 = tf.keras.layers.Dense(2,activation='linear')(reshape2)

    # Recurrent1 creates s_t
    h0 = np.zeros((1,w_g+1))
    h0[0,0] = 1
    kh0 = tf.keras.backend.constant(h0)
    recurrent1 = tf.keras.layers.GRU(w_g+1, activation='linear', recurrent_activation='linear', return_sequences=True, reset_after=False)(dense1, initial_state=kh0)
    
    merge1 = tf.keras.layers.Concatenate(axis=-1)([dense1, recurrent1])
    dense2 = tf.keras.layers.Dense(w_g, activation='relu')(merge1)
    merge2 = tf.keras.layers.Concatenate(axis=-1)([dense1, dense2])
    recurrent2 = tf.keras.layers.GRU(w_g, activation='linear', recurrent_activation='relu', return_sequences=False, reset_after=False)(merge2)
    
    gru_model = tf.keras.Model(inputs=inputs, outputs=recurrent2, name='gru_model')
    gru_model.compile(loss='mean_squared_error', metrics=['accuracy'])

    # Weight assignments
    w0 = np.array([[-1,  1, -1,  1,  1,  1,  1,  1,  1,  1],
                   [ 1, -1, -1,  1,  1, -1, -1,  1, -1, -1],
                   [-1,  1,  1,  1,  1, -1,  1,  1,  1,  1],
                   [-1, -1, -1,  1,  1,  1,  1, -1,  1, -1],
                   [ 1,  1, -1, -1, -1, -1,  1,  1, -1,  1],
                   [-1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1],
                   [ 1,  1,  1,  1,  1,  1, -1,  1,  1, -1],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1]])
    w1 = w0.reshape(3,3,1,10)
    b0 = np.array([-2, -3, -4, -6, -7, -4, -6, -4, -6, -4])
    gru_model.layers[2].set_weights([w1, b0])

    w2 = np.array([[1,1],[2,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1],[9,1],[10,1]])
    b1 = np.zeros(2)
    gru_model.layers[5].set_weights([w2,b1])

    # Recurrent1
    w3 = np.zeros((2, 3*(w_g+1)))
    w3[1, 0:(w_g+1)] = -np.ones((1, w_g+1)) # z_t = 1 - I(x>0)
    w4 = np.zeros((w_g+1, 3*(w_g+1)))
    for j in range(w_g):
        w4[j, 2*(w_g+1)+j+1] = 1 # Set W_h to be right shift matrix
    b2 = np.zeros(3*(w_g+1))
    b2[0:2*(w_g+1)] = np.ones(2*(w_g+1)) # z_t = 1 - I(x>0), r_t = 1
    gru_model.layers[6].set_weights([w3, w4, b2])

    w5 = np.zeros((3+w_g, w_g))
    w5[1,:] = np.ones(w_g)
    w5[3:(3+w_g),:] = np.eye(w_g)
    b3 = -np.ones(w_g)
    gru_model.layers[8].set_weights([w5, b3])

    w6 = np.zeros((2+w_g, 3*w_g))    
    w6[0,(2*w_g):(3*w_g)] = np.ones(w_g) # r^t = 1, h^_t = x_t
    w6[2:(2+w_g),0:w_g] = -np.eye(w_g) # z_t = 1 - I(s=j and x>0)
    w7 = np.zeros((w_g, 3*w_g))
    b4 = np.zeros(3*w_g)
    b4[0:w_g] = np.ones(w_g) # z_t = 1 - I(s=j and x>0)
    gru_model.layers[10].set_weights([w6, w7, b4])

    return gru_model


def make_vocab(file):
    """
    Make a vocabulary (for lookup layers), from a file. The intention is to use
    words_3x3.txt.

    file : .txt file
    """
    src_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(src_dir, file)
    words_3x3 = [l.replace('\n', '').lower() for l in open(file_path, 'r').readlines()]
    letters = ["i","y","j","c","o","l","h","t","u","x"]
    layer1 = tf.keras.layers.StringLookup(vocabulary=letters)
    numbers_3x3 = []
    for word in words_3x3:
        digits = layer1(list(word)).numpy()
        word_sum = sum(digits*np.array([11]*len(digits))**np.arange(len(digits)))
        numbers_3x3.append(word_sum)
    return words_3x3, numbers_3x3


def make_2vocab(file):
    """
    Make a vocabulary (for lookup layers), from a file. The intention is to use
    words_3x3.txt. This assumes the file is in the current folder and works in
    the Jupyter notebook.

    file : .txt file
    """
    words_3x3 = [l.replace('\n', '').lower() for l in open(file, 'r').readlines()]
    letters = ["i","y","j","c","o","l","h","t","u","x"]
    layer1 = tf.keras.layers.StringLookup(vocabulary=letters)
    numbers_3x3 = []
    for word in words_3x3:
        digits = layer1(list(word)).numpy()
        word_sum = sum(digits*np.array([11]*len(digits))**np.arange(len(digits)))
        numbers_3x3.append(word_sum)
    return words_3x3, numbers_3x3


def make_full_model(h, w, file='..\\words_3x3.txt'):
    """
    Create a model that recognises words within a grid of height h and width w.
    Adds lookup layers to return an actual character string.
    
    The output is a tensor containing a word taken from the vocabulary created by
    the input file. It is assumed the grid contains a single word written from 
    left to right.

    h : integer, height of input image
    w : integer, width of input image
    file : .txt file
    """
    w_g = int(np.floor(w/3))
    inputs = tf.keras.Input(shape=(h, w))
    reshape1 = tf.keras.layers.Reshape((h, w, 1))(inputs)
    convlayer = tf.keras.layers.Conv2D(10, [3, 3], [1, 1], padding='valid', activation='relu')(reshape1)
    pooling = tf.keras.layers.MaxPooling2D(pool_size=(h,3), strides=(h,3), padding='same')(convlayer)
    reshape2 = tf.keras.layers.Reshape((w_g, 10))(pooling)
    dense1 = tf.keras.layers.Dense(2,activation='linear')(reshape2)
    h0 = np.zeros((1,w_g+1))
    h0[0,0] = 1
    kh0 = tf.keras.backend.constant(h0)
    recurrent1 = tf.keras.layers.GRU(w_g+1, activation='linear', recurrent_activation='linear', return_sequences=True, reset_after=False)(dense1, initial_state=kh0) 
    merge1 = tf.keras.layers.Concatenate(axis=-1)([dense1, recurrent1])
    dense2 = tf.keras.layers.Dense(w_g, activation='relu')(merge1)
    merge2 = tf.keras.layers.Concatenate(axis=-1)([dense1, dense2])
    recurrent2 = tf.keras.layers.GRU(w_g, activation='linear', recurrent_activation='relu', return_sequences=False, reset_after=False)(merge2)

    # Convert the GRU output to a single number and then to a word using lookup layers   
    words_3x3, numbers_3x3 = make_vocab(file=file)
    dense3 = tf.keras.layers.Dense(1, activation='linear')(recurrent2)
    lookup1 = tf.keras.layers.IntegerLookup(vocabulary=numbers_3x3)(dense3)
    lookup2 = tf.keras.layers.StringLookup(vocabulary=words_3x3, invert=True)(lookup1)

    full_model = tf.keras.Model(inputs=inputs, outputs=lookup2, name='full_model')
    full_model.compile(loss='mean_squared_error', metrics=['accuracy'])

    # Use layers from GRU model up to recurrent2
    w0 = np.array([[-1,  1, -1,  1,  1,  1,  1,  1,  1,  1],
                   [ 1, -1, -1,  1,  1, -1, -1,  1, -1, -1],
                   [-1,  1,  1,  1,  1, -1,  1,  1,  1,  1],
                   [-1, -1, -1,  1,  1,  1,  1, -1,  1, -1],
                   [ 1,  1, -1, -1, -1, -1,  1,  1, -1,  1],
                   [-1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1],
                   [ 1,  1,  1,  1,  1,  1, -1,  1,  1, -1],
                   [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1]])
    w1 = w0.reshape(3,3,1,10)
    b0 = np.array([-2, -3, -4, -6, -7, -4, -6, -4, -6, -4])
    full_model.layers[2].set_weights([w1, b0])

    w2 = np.array([[1,1],[2,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1],[9,1],[10,1]])
    b1 = np.zeros(2)
    full_model.layers[5].set_weights([w2,b1])

    # Recurrent1
    w3 = np.zeros((2, 3*(w_g+1)))
    w3[1, 0:(w_g+1)] = -np.ones((1, w_g+1)) # z_t = 1 - I(x>0)
    w4 = np.zeros((w_g+1, 3*(w_g+1)))
    for j in range(w_g):
        w4[j, 2*(w_g+1)+j+1] = 1 # Set W_h to be right shift matrix
    b2 = np.zeros(3*(w_g+1))
    b2[0:2*(w_g+1)] = np.ones(2*(w_g+1)) # z_t = 1 - I(x>0), r_t = 1
    full_model.layers[6].set_weights([w3, w4, b2])

    w5 = np.zeros((3+w_g, w_g))
    w5[1,:] = np.ones(w_g)
    w5[3:(3+w_g),:] = np.eye(w_g)
    b3 = -np.ones(w_g)
    full_model.layers[8].set_weights([w5, b3])

    w6 = np.zeros((2+w_g, 3*w_g))    
    w6[0,(2*w_g):(3*w_g)] = np.ones(w_g) # r^t = 1, h^_t = x_t
    w6[2:(2+w_g),0:w_g] = -np.eye(w_g) # z_t = 1 - I(s=j and x>0)
    w7 = np.zeros((w_g, 3*w_g))
    b4 = np.zeros(3*w_g)
    b4[0:w_g] = np.ones(w_g) # z_t = 1 - I(s=j and x>0)
    full_model.layers[10].set_weights([w6, w7, b4])

    w8 = np.array([11]*w_g)**np.arange(w_g)
    w8 = w8.reshape(w_g, 1)
    b5 = np.zeros(1)
    full_model.layers[11].set_weights([w8, b5])

    return full_model