import sys
import pickle
import numpy as np

from uuid import uuid4
from collections import namedtuple, deque 
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from itertools import takewhile, islice
from argparse import ArgumentParser

from snakenet.game import Game
from snakenet.model_player import get_model_prediction_idx, VALID_MOVES, get_model, get_future_rewards
from snakenet.game_constants import DOWN, UP, LEFT, RIGHT, NUM_ROWS, NUM_COLUMNS, LoseException
from snakenet.snake_printer import print_plane

TRANSITION_MEMORY = 1000000 # Number of 'inter-frames' to remember
DISCOUNT_FACTOR_GAMMA = 0.99
N_VALID_MOVES = len(VALID_MOVES)

ALIVE_POINTS = 0
EATING_POINTS = 1
DYING_POINTS = -1

RANDOM_SAMPLE_BATCH_SIZE = 25 


# Input 1  = (None, 1, r, c)
# Output 1 = (None, 16, r, c) w/ border_mode='same'
# Input 2  = (None, 16, r, c)
# Output 2 = (None, 32, r, c) w/ border_mode='same'
CONV_CONFIG = [ (16, (3, 3), 'same')
              , (32, (9, 9), 'same')
              ]

# 1 is number of channels (this is grayscale).
input_shape = (1, NUM_ROWS, NUM_COLUMNS)

def get_random_move_probability(cur_epoch):
    if cur_epoch > 1e6:
        return 0.1
    else:
        return 0.9 * (1 - (cur_epoch / 1e6)) + 0.1

def get_random_move_idx():
    return np.random.choice(4, size=1)[0] 

def construct_model():
    # The snake state plane model.
    input_plane = Input(shape=input_shape)

    plane_net = Sequential()
    # Loop over convolutional layers and add them.
    for idx, (filters, kernel_size, border_mode) in enumerate(CONV_CONFIG):
        kwargs = { 'border_mode': border_mode, 'activation':'relu' }
        if idx == 0:
            kwargs['input_shape'] = input_shape
        # Adding includes conv layer and activation function.     
        plane_net.add(Convolution2D(filters, kernel_size[0], kernel_size[1], **kwargs))
    plane_net.add(Flatten())

    # Encode action.
    action_input = Input(shape=(N_VALID_MOVES,))

    # Encode inputs.
    encoded_plane = plane_net(input_plane)

    # Merge encoded values together.
    merged = merge([action_input, encoded_plane], mode='concat')

    # The reward predicting network (filters is from the last iteration of the loop).
    final_shape = int( NUM_ROWS * NUM_COLUMNS * filters + len(VALID_MOVES) )
    last_stage = Sequential()
    last_stage.add(Dense(256, input_shape=(final_shape,)))
    last_stage.add(Activation('relu'))
    last_stage.add(Dense(1))
    last_stage.add(Activation('linear'))

    output = last_stage(merged)

    model = Model(input=[action_input, input_plane], output=output)

    # Compile the model.
    model.compile(loss='mse', optimizer='nadam')

    return model

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
        self.model.fit(inputs, targets, nb_epoch=1, verbose=0, **kwargs)

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

def trigger_and_return_action(game, sequence, model):
    p = np.random.random(1)
    if p < get_random_move_probability(model.n_epochs):
        move_idx = get_random_move_idx()
    else:
        move_idx = get_model_prediction_idx(game, model)
    game.keypress(VALID_MOVES[move_idx])
    return move_idx

def get_value_from_sample(sample, model):
    """ Calculate the Q value for this transition """
    if sample.terminal:
        return sample.reward
    else:
        return np.max(get_future_rewards(sample.new_state, model)) * DISCOUNT_FACTOR_GAMMA

def do_learning(model, sequence):
    samples = sequence.random_sample(RANDOM_SAMPLE_BATCH_SIZE)
    images = []
    actions = []
    values = []
    for sample in samples:
        values.append(get_value_from_sample(sample, model))
        images.append(sample.old_state[np.newaxis,...])
        action_array = np.zeros(len(VALID_MOVES))
        action_array[sample.action] = 1
        actions.append(action_array)

    values = np.array(values)
    images = np.array(images)
    actions = np.array(actions)

    model.fit([actions, images], values, batch_size=RANDOM_SAMPLE_BATCH_SIZE)

def do_game(sequence, model, game_number):
    game = Game(graphical=False)
    while True: # Iterate moves.
        sys.stdout.flush()
        old_state = game.state.plane.copy()
        old_times_eaten = game.state.times_eaten
        # Guess that we don't die.
        terminal = False
        reward = ALIVE_POINTS 
        try:
            action = trigger_and_return_action(game, sequence, model)
            game.move() # Actually trigger movement.
            new_times_eaten = game.state.times_eaten
            if new_times_eaten > old_times_eaten: 
                reward = EATING_POINTS # Eating is really good.
        except LoseException as e:
            reward = DYING_POINTS # Game over is really bad.
            terminal = True
        new_state = game.state.plane.copy()

        #print_plane(new_state) 

        # Save this transition.
        sequence.record_transition(old_state, action, reward, new_state, terminal)

        # Begin the learning phase
        if sequence.is_ready_to_train():
            do_learning(model, sequence)
        if terminal:
            sequence.finish_game()
            return game.state.moves, game.state.times_eaten

def do_execution(sequence, model, args):
    mode = 'a' if args.restart else 'w'
    with open(args.logfile, mode) as log:
        if mode == 'w':
            log.write('epochs,games,moves,food,p_random')
            log.write('\n')
        print('Beginning Execution')
        game_number = 0
        moves_history = []
        food_history = []
        while True: # Iterate game.
            moves, foods = do_game(sequence, model, game_number)
            moves_history.append(moves)
            food_history.append(foods)
            game_number += 1
            if game_number % 50 == 0:
                avg_moves = np.mean(moves_history)
                avg_food = np.mean(food_history)
                rand_move_prob = get_random_move_probability(model.n_epochs)

                log.write('{},{},{:0.4f},{:0.4f},{:0.4f}'.format(model.n_epochs, game_number, avg_moves, avg_food, rand_move_prob))
                log.write('\n')
                log.flush()

                print('Avg Moves = {:0.2f}'.format(avg_moves))
                print('Avg food = {:0.2f}'.format(avg_food))
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
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--logfile', default='training.csv')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.restart == True:
        model = get_model()
        sequence = get_sequence()
    else:
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
