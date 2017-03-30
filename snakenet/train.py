import sys
import time
import pickle
import numpy as np

from warnings import catch_warnings
from uuid import uuid4
from collections import namedtuple, deque 
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.optimizers import Nadam, RMSprop
from itertools import takewhile, islice
from argparse import ArgumentParser

from snakenet.game import Game
from snakenet.model_player import get_model_prediction_idx, VALID_MOVES, get_model, get_action_values 
from snakenet.game_constants import DOWN, UP, LEFT, RIGHT, NUM_ROWS, NUM_COLUMNS, LoseException
from snakenet.snake_printer import print_plane
from snakenet.draw import draw_game

TRANSITION_MEMORY = int(1e6) # Number of 'inter-frames' to remember
DISCOUNT_FACTOR_GAMMA = 0.95
N_VALID_MOVES = len(VALID_MOVES)

ALIVE_POINTS = 0
EATING_POINTS = 1
USELESS_MOVE_POINTS = -0.1
DYING_POINTS = -1

RANDOM_SAMPLE_BATCH_SIZE = 25 


# Filters, filter size, padding.
CONV_CONFIG = [ (32, (3, 3), 'same') 
              #, (32, (6, 6), 'same')
              ]
              
POOL_SIZE = (2, 2) 

# 1 is number of channels (this is grayscale).
input_shape = (1, NUM_ROWS, NUM_COLUMNS)

# Start at 1.0. Decline linearly for NUM_DECLINING_EPOCHS towards RANDOM_END_P.
NUM_DECLINING_EPOCHS = 1e6
RANDOM_END_P = 0.05
def get_random_move_probability(cur_epoch):
    if cur_epoch > NUM_DECLINING_EPOCHS:
        return RANDOM_END_P
    else:
        starting_p = 1. - RANDOM_END_P
        decline_amount = (1. - (cur_epoch / NUM_DECLINING_EPOCHS)) 
        return starting_p * decline_amount + RANDOM_END_P

def get_random_move_idx():
    return np.random.choice(4, size=1)[0] 

def construct_model():
    action_value_net = Sequential()
    # Loop over convolutional layers and add them.
    for idx, (filters, kernel_size, padding) in enumerate(CONV_CONFIG):
        kwargs = { 'padding': padding, 'use_bias': False }
        if idx == 0:
            kwargs['input_shape'] = input_shape
        # Adding includes conv layer and activation function.     
        action_value_net.add(Convolution2D(filters, kernel_size, **kwargs))
        action_value_net.add(PReLU())
        action_value_net.add(BatchNormalization(axis=1))

    # Trying to get translation invariance.
    action_value_net.add(MaxPooling2D(pool_size=POOL_SIZE, strides=POOL_SIZE, padding="same"))
    # Flatten out for final dense layer.
    action_value_net.add(Flatten())

    msg = 'Error: Pool size does not evenly divide image size'
    assert float(NUM_ROWS) / POOL_SIZE[0] % 1 == 0., msg
    assert float(NUM_COLUMNS) / POOL_SIZE[1] % 1 == 0., msg 

    action_value_net.add(Dense(512, use_bias=False))
    action_value_net.add(BatchNormalization(axis=1))
    action_value_net.add(Activation('relu'))
    action_value_net.add(Dense(len(VALID_MOVES)))
    action_value_net.add(Activation('linear'))

    # Compile the model.
    optimizer = RMSprop()
    action_value_net.compile(loss='mse', optimizer=optimizer)

    return action_value_net 

class QNetwork(object):
    def __init__(self):
        self.model = construct_model()
        self.n_epochs = 0
        self.avg_moves = [] 
        self.avg_food = []
        self.n_games = []
        self.epoch_counter = []

    def save_performance(self, moves, foods, games):
        self.avg_moves.append(moves)
        self.avg_food.append(foods)
        self.n_games.append(games)
        self.epoch_counter.append(self.n_epochs)

    def fit(self, inputs, targets, **kwargs):
        self.n_epochs += 1
        self.model.fit(inputs, targets, epochs=1, verbose=0, **kwargs)

    def predict(self, inputs):
        return self.model.predict(inputs)

Transition = namedtuple('Transition', ['old_state', 'action', 'reward', 'new_state', 'terminal'])

class TransitionSequence(object):
    def __init__(self):
        self.transitions = deque(maxlen=TRANSITION_MEMORY) 
        self.next_transitions = []

    def random_sample(self, sample_size):
        idxs = np.random.choice(len(self.transitions), size=sample_size) 

        samples = []
        for idx in idxs:
            samples.append(self.transitions[idx])

        return samples

    def record_transition(self, old_state, action, reward, new_state, terminal):
        self.next_transitions.append(Transition(old_state, action, reward, new_state, terminal))

    def is_ready_to_train(self):
        return len(self.transitions) > 0

    def finish_game(self):
        for nt in self.next_transitions:
            self.transitions.append(nt)

RANDOM_MOVE_ACTION_VALUES = (None,)*4

def trigger_and_return_action(game, sequence, model):
    p = np.random.random(1)
    if p < get_random_move_probability(model.n_epochs):
        move_idx = get_random_move_idx()
        action_values = RANDOM_MOVE_ACTION_VALUES
    else:
        move_idx, action_values = get_model_prediction_idx(game, model)
    was_valid = game.keypress(VALID_MOVES[move_idx])
    return move_idx, was_valid, action_values

def get_value_from_sample(sample, model):
    """ Calculate the Q value for this transition """
    if sample.terminal:
        return sample.reward
    else:
        # Note: This is the Bellman equation!
        future_reward = np.max(get_action_values(sample.new_state, model))
        discounted_future_reward = future_reward * DISCOUNT_FACTOR_GAMMA
        return sample.reward + discounted_future_reward

def do_learning(model, sequence):
    samples = sequence.random_sample(RANDOM_SAMPLE_BATCH_SIZE)
    images = []
    values = []
    actions = []
    for sample in samples:
        values.append(get_value_from_sample(sample, model))
        images.append(sample.old_state[np.newaxis,...])
        actions.append(sample.action)

    predictions = model.predict(np.array(images))

    # Update desired output _only_ for the actions that we actually observed.
    # Leave the other outputs set to the predicted values (Assume unobserved 
    # action values were predicted correctly). This clever assumption allows 
    # us to update a network which predicts all action values simultaneously 
    # even though we don't observe all possible actions.
    desired_output = predictions
    rows = np.arange(len(desired_output))
    cols = actions
    desired_output[rows, cols] = values

    images = np.array(images)

    model.fit([images], desired_output, batch_size=RANDOM_SAMPLE_BATCH_SIZE)

def do_game(sequence, model, game_number, visible, sleep_ms):
    game = Game(graphical=visible)
    if visible:
        draw_game(game)

    value_history = []
    while True: # Iterate moves.
        sys.stdout.flush()
        old_state = game.state.plane.copy()
        old_times_eaten = game.state.times_eaten
        # Guess that we don't die.
        terminal = False
        reward = ALIVE_POINTS 
        try:
            action, was_valid_action, action_values = trigger_and_return_action(game, sequence, model)
            game.move(action_values=action_values) # Actually trigger movement.
            if action_values is not RANDOM_MOVE_ACTION_VALUES:
                value_history.append(np.max(action_values))
            if visible:
                draw_game(game)
                time.sleep(sleep_ms / 1000.0)
            new_times_eaten = game.state.times_eaten
            if not was_valid_action:
                reward += USELESS_MOVE_POINTS
            if new_times_eaten > old_times_eaten: 
                reward += EATING_POINTS # Eating is really good.
        except LoseException as e:
            reward = DYING_POINTS # Game over is really bad.
            terminal = True
        new_state = game.state.plane.copy()

        # Save this transition.
        sequence.record_transition(old_state, action, reward, new_state, terminal)

        # Begin the learning phase
        if sequence.is_ready_to_train():
            do_learning(model, sequence)
        if terminal:
            sequence.finish_game()
            return game.state.moves, game.state.times_eaten, np.mean(value_history)

def do_execution(sequence, model, args):
    mode = 'w' if args.fresh else 'a'
    with open(args.logfile, mode) as log:
        if mode == 'w':
            log.write('epochs,games,moves,food,mean_val,p_random')
            log.write('\n')
        print('Beginning Execution')
        game_number = 0
        moves_history = []
        food_history = []
        while True: # Iterate game.
            moves, foods, mean_val = do_game(sequence, model, game_number, args.visible, args.sleep_ms)
            moves_history.append(moves)
            food_history.append(foods)
            game_number += 1
            if game_number % 50 == 0:
                avg_moves = np.mean(moves_history)
                avg_food = np.mean(food_history)
                rand_move_prob = get_random_move_probability(model.n_epochs)

                fmt = '{},{},{:0.4f},{:0.4f},{:0.4f},{:0.4f}'
                log.write(fmt.format(model.n_epochs, game_number, avg_moves, avg_food, mean_val, rand_move_prob))
                log.write('\n')
                log.flush()

                print('Avg Moves = {:0.2f}'.format(avg_moves))
                print('Avg food = {:0.2f}'.format(avg_food))
                print('Mean AV = {:0.2f}'.format(mean_val))
                print('(P) Random = {:0.2f}'.format(rand_move_prob))

                model.save_performance(avg_moves, avg_food, game_number)

                moves_history = []
                food_history = []

def get_sequence():
    with open('sequence.pkl', 'rb') as f:
        sys.setrecursionlimit(25000)
        sequence = pickle.load(f)
    return sequence

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--logfile', default='log_training.csv')
    parser.add_argument('--visible', action='store_true')
    sleep_help = "Time to wait between updates if --visible"
    parser.add_argument('--sleep-ms', default=0, type=int, help=sleep_help)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.fresh == False:
        print('Continuing training')
        model = get_model()
        sequence = get_sequence()
    else:
        print('Beginning new training')
        model = QNetwork()
        sequence = TransitionSequence()
    try:
        do_execution(sequence, model, args)
    except KeyboardInterrupt:
        print()
        print("KeyboardInterrupt Received. Writing Model.")

        sys.setrecursionlimit(25000)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('sequence.pkl', 'wb') as f:
            pickle.dump(sequence, f)

if __name__ == '__main__':
    main()
