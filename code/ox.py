from re import I
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import os
import pandas as pd
from itertools import combinations

# Global variables to help create the perfect player in suggest_move()
board = range(9)

wins = np.array([
    [1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 1, 0, 0]
])

ee = np.eye(9)
eee = np.zeros([9, 2, 9, 2])

def make_eee():
    global eee
    for i in range(9):
        for j in range(2):
            eee[i, j, i, j] = 1

make_eee()

corners = np.array([1, 0, 1, 0, 0, 0, 1, 0, 1])
edges   = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])

opposite = [8, 7, 6, 5, 4, 3, 2, 1, 0]

empty_board = np.zeros([9, 2])


def add_o(pq, i): pq[i, 0] = 1

def add_x(pq, i): pq[i, 1] = 1


def has_win(pq, player=None):
    """
    Checks whether a player has won the game in board pq.
    """
    w = [np.max(np.matmul(wins, pq[:, j])) >= 3 for j in [0, 1]]
    if player is None:
        return w[0] or w[1]
    else:
        return w[player]


def plays(pq, player=0):
    """
    Returns the available moves to a player in board pq.
    """
    n = []
    for i in range(9):
        if pq[i, 0] == 0 and pq[i, 1] == 0:
            n.append(i)
    return n


def near_wins(pq, player=0):
    """
    Returns the moves a player can play to immediately win in board pq.
    """
    n = []
    for i in range(9):
        if pq[i, 0] == 0 and pq[i, 1] == 0 and has_win(pq + eee[i, player], player):
            n.append(i)
    return n


def forks(pq, player=0):
    """
    Returns the moves a player can play to create two near-wins in board pq.
    """
    f = []
    for i in range(9):
        if pq[i, player] == 0 and pq[i, 1-player] == 0 and \
                len(near_wins(pq + eee[i, player], player)) > 1:
            f.append(i)
    return f


def swap(pq):
    return np.array([[pq[i, 1-j] for j in range(2)] for i in range(9)])


def random_move(pq, player=0):
    """
    Returns a random move out of the available positions in board pq.
    """
    if has_win(pq) or len(plays(pq)) == 0:
        return None
    return np.random.choice(plays(pq, player))


def suggest_move(pq, player):
    """
    Returns the best move for the player in board pq. This has been modified to
    start in a random corner if given an empty board.
    """
    if player == 0:
        pq0 = pq.copy()
        qp0 = swap(pq0)
    else:
        qp0 = pq.copy()
        pq0 = swap(qp0)
    if has_win(pq0) or len(plays(pq0)) == 0:
        return None
    n = near_wins(pq0)
    if len(n) > 0: return n[0]
    m = near_wins(qp0)
    if len(m) > 0: return m[0]
    f = forks(pq0)
    if len(f) > 0: return f[0]
    g = forks(qp0)
    if len(g) == 1: return g[0]
    for i in plays(pq0):
        pq1 = pq0 + eee[i, 0]
        qp1 = swap(pq1)
        if len(near_wins(pq1)) > 0 and len(forks(qp1)) == 0:
            return i
    if sum(sum(pq0)) == 0: return np.random.choice(np.array([0,2,6,8]))
    if pq0[4, 0] == 0 and pq0[4, 1] == 0: return 4
    for i in [0, 2, 6, 8]:
        if pq0[i, 0] == 0 and pq0[i, 1] == 0 and pq0[8-i, 1] == 1:
            return i
    for i in [0, 2, 6, 8, 1, 3, 5, 7]:
        if pq0[i, 0] == 0 and pq0[i, 1] == 0:
            return i
    return None


def make_all_states(SHUFFLE_SIZE=100, BATCH_SIZE=32):
    """
    Creates the set of all board states and the corresponding best move.
    """
    all_states = []
    states = [[empty_board.copy()]]
    for i in range(1,9):
        states.append([])
        for s in states[i-1]:
            p = plays(s)
            k = 8
            while s[k, i % 2] == 0 and k >= 0: 
                k = k-1
            for j in p:
                s1 = s + eee[j, i % 2]
                if j > k and not has_win(s1):
                    states[i].append(s1)
        all_states += states[i]
    
    all_states_suggestions = list(map(
        lambda pq: ee[suggest_move(pq, 1)], 
        all_states
    ))

    all_dataset = tf.data.Dataset.from_tensor_slices((all_states, all_states_suggestions))
    all_dataset = all_dataset.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)

    return states, all_states, all_states_suggestions, all_dataset



def make_simple_model(p=20, q=20, r=0.001):
    inputs = tf.keras.Input(shape=(9, 2))
    reshape = tf.keras.layers.Reshape((18,))(inputs)
    hidden0 = tf.keras.layers.Dense(p, activation='relu')(reshape)
    hidden1 = tf.keras.layers.Dense(q, activation='relu')(hidden0)
    outputs = tf.keras.layers.Dense(9, activation='softmax')(hidden1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="model")
    opt = tf.keras.optimizers.Adam(learning_rate=r)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def make_res_model(p=10, q=10, r=0.01, residual_train=True, fix_pen=20):
    inputs = tf.keras.Input(shape=(9, 2))
    reshape = tf.keras.layers.Reshape((18,))(inputs)
    hidden0 = tf.keras.layers.Dense(p, activation='relu')(reshape)
    hidden1 = tf.keras.layers.Dense(q, activation='relu')(hidden0)
    outputs1 = tf.keras.layers.Dense(9, activation='linear')(hidden1)
    outputs2 = tf.keras.layers.Dense(9, activation='linear', trainable=residual_train)(reshape)
    resout = tf.keras.layers.Add()([outputs1,outputs2])
    resout = tf.keras.layers.Activation("softmax")(resout)
    res_model = tf.keras.Model(inputs=inputs, outputs=resout, name="model")
    opt = tf.keras.optimizers.Adam(learning_rate=r)
    res_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    w1 = np.zeros((18,9))
    for i in range(9):
        w1[2*i,i] = 1
        w1[2*i+1,i] = 1
    w1 = -fix_pen*w1
    b1 = np.zeros(9)
    if not residual_train:
        res_model.layers[5].set_weights([w1,b1])
    
    return res_model


# This function trains the model to replicate the behaviour of the
# suggest_move() function
def train_model_suggestions(M, epochs, batch_size=32):
    states, all_states, all_states_suggestions, all_dataset = make_all_states(SHUFFLE_SIZE=100, BATCH_SIZE=batch_size)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, verbose=0)
    return M.fit(all_dataset, epochs=epochs, callbacks=[callback], verbose=0)

# This function applies the model to a game state pq to generate
# a probability distribution on the board positions, and then 
# selects a position randomly in accordance with that distribution.
# (If the model is not well-trained then the move may be illegal, 
# resulting in an immediate loss.)
def model_move(M, pq, player=0, argmax=False):
    if player == 0:
        pq0 = pq.copy()
    else:
        pq0 = swap(pq.copy())
    prob = M.predict(pq0.reshape(1, 9, 2), verbose=0)[0]
    choices = range(9)
    if argmax: return choices[np.argmax(prob)]
    else: return np.random.choice(choices, p=prob)
    


# This function displays the given game state as an image 
# using matplotlib
def show_board(pq, winner=None):
    col_o = 'red'
    col_x = 'blue'
    if winner == 0: col_x = 'lightblue'
    if winner == 1: col_o = 'lightcoral'
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_axis_off()
    ax.set_xlim((0, 3))
    ax.set_ylim((0, 3))
    ax.add_line(matplotlib.lines.Line2D([1, 1], [0, 3], color='grey'))
    ax.add_line(matplotlib.lines.Line2D([2, 2], [0, 3], color='grey'))
    ax.add_line(matplotlib.lines.Line2D([0, 3], [1, 1], color='grey'))
    ax.add_line(matplotlib.lines.Line2D([0, 3], [2, 2], color='grey'))
    for i in range(3):
        for j in range(3):
            k = i+3*j
            if pq[k, 0] == 1:
                ax.add_patch(matplotlib.patches.Circle([i + 0.5, j + 0.5], 0.3, color=col_o, fill=False))
            if pq[k, 1] == 1:
                ax.add_line(matplotlib.lines.Line2D([i + 0.2, i + 0.8], [j + 0.2, j + 0.8], color=col_x))
                ax.add_line(matplotlib.lines.Line2D([i + 0.2, i + 0.8], [j + 0.8, j + 0.2], color=col_x))


# This function prints an ASCII art representation of the specified game state
def show_board_ascii(pq, winner=None):
    sym_o = 'O'
    sym_x = 'X'
    if winner == 0: sym_x = 'x'
    if winner == 1: sym_o = 'o'
    s = '---\n'
    for j in range(3):
        for i in range(3):
            k = i+3*(2-j)
            t = '#'
            if pq[k, 0] == 1: t = sym_o
            if pq[k, 1] == 1: t = sym_x
            s = s + t
        s = s + '\n'
    s = s + '---'
    print(s)


# This function plays a game.
# It assumes that player_o and player_x are functions like
# random_move, suggest_move, input_move and model_move:
# they should accept a game state and player index, and 
# return a board position.  By default play_game() returns
# the final board position and the index of the winner 
# (0 for O, 1 for X).  If the return_moves argument is true
# it also returns a list moves = [moves_o, moves_x] 
# specifying the moves taken in order.
def play_game(player_o, player_x, return_moves=False):
    pq = empty_board.copy()
    winner = None
    # edit: track illegal moves
    illegal = None
    moves_o = []
    moves_x = []
    while True:
        if len(plays(pq)) == 0:
            break
        i = player_o(pq, 0)
        moves_o.append(i)
        if i is None or pq[i, 0] == 1 or pq[i, 1] == 1:
            winner = 1
            illegal = 0
            break
        add_o(pq, i)
        if has_win(pq, 0):
            winner = 0
            break
        if has_win(pq, 1):
            winner = 1
            break
        if len(plays(pq)) == 0:
            break
        i = player_x(pq, 1)
        moves_x.append(i)
        if i is None or pq[i, 0] == 1 or pq[i, 1] == 1:
            winner = 0
            illegal = 1
            break
        add_x(pq, i)
        if has_win(pq, 0):
            winner = 0
            break
        if has_win(pq, 1):
            winner = 1
            break
    if return_moves:
        return pq, winner, illegal, [moves_o, moves_x]
    else:
        return pq, winner, illegal



# Now try using the same structure but set the residual connections first
# and prevent them from being trained

def play_match(player_o, player_x, bestof=100):
    scores = [0, 0, 0]
    illegals = [0, 0]

    for game in range(bestof):
        pq, winner, illegal = play_game(player_o, player_x)
        if winner == 0: scores[1] += 1
        elif winner ==1: scores[2] += 1
        else: scores[0] += 1
        if illegal == 0: illegals[0] += 1
        elif illegal == 1: illegals[1] += 1
        #print("Game " + str(game+1) + "/" + str(bestof)+ " done.")

    return scores, illegals

#make_model(p=20, q=20, r=0.01)
#train_model_suggestions()
#play_match(model_move, suggest_move)
#scores, illegals = play_match(res_model_move, suggest_move)


# here we can achieve 99% performance with res_model using
# fixed residual connections and p=q=10 for the rest of the network

# or letting the network train the residual connections and
# results in approx -20 to -40 weights on diagonals and not much else

def test_one_model(M, no_games=100):
    def test_model_move(pq, player=0, argmax=False):
        return model_move(M=M, pq=pq, player=0, argmax=False)

    scores, illegals = play_match(test_model_move, suggest_move, bestof=no_games)
    print(f'Without argmax, versus perfect player:')
    print(f'Draws: {scores[0]}, \nWins: {scores[1]}, \nLosses: {scores[2]}, \nLosses due to illegal moves: {illegals[0]}')

    def test_model_move_argmax(pq, player=0, argmax=True):
        return model_move(M=M, pq=pq, player=0, argmax=True)

    scores, illegals = play_match(test_model_move_argmax, suggest_move, bestof=no_games)
    print(f'\nWith argmax, versus perfect player:')
    print(f'Draws: {scores[0]}, \nWins: {scores[1]}, \nLosses: {scores[2]}, \nLosses due to illegal moves: {illegals[0]}')

    scores, illegals = play_match(test_model_move_argmax, random_move, bestof=no_games)
    print(f'\nWith argmax, versus random player:')
    print(f'Draws: {scores[0]}, \nWins: {scores[1]}, \nLosses: {scores[2]}, \nLosses due to illegal moves: {illegals[0]}')



def test_models(models):
    results_data = []
    for M in models:
        model_result = []
        def test_model_move(pq, player=0, argmax=False):
            return model_move(M=M, pq=pq, player=0, argmax=False)
        score, illegal = play_match(test_model_move, suggest_move)
        model_result.append(score + illegal)
        def test_model_move_argmax(pq, player=0, argmax=True):
            return model_move(M=M, pq=pq, player=0, argmax=True)
        score, illegal = play_match(test_model_move_argmax, suggest_move)
        model_result.append(score + illegal)
        score, illegal = play_match(test_model_move_argmax, random_move)
        model_result.append(score + illegal)
        results_data.append(model_result)

    return results_data