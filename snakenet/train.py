import sys
import pickle
import numpy as np

from uuid import uuid4
from collections import namedtuple, deque 
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from itertools import takewhile, islice

from snakenet.game import Game
from snakenet.model_player import get_model_prediction_idx, VALID_MOVES 
from snakenet.game_constants import DOWN, UP, LEFT, RIGHT, NUM_ROWS, NUM_COLUMNS, LoseException

TRANSITION_MEMORY = 100000 # Number of 'inter-frames' to remember
DISCOUNT_FACTOR_GAMMA = 0.8
RANDOM_MOVE_PROBABILITY = 1.0 / 5
N_VALID_MOVES = len(VALID_MOVES)

ALIVE_POINTS = 0
EATING_POINTS = 1
DYING_POINTS = -1

RANDOM_SAMPLE_BATCH_SIZE = 50 

nb_filters = 32 # Number of distinct convolutional filters to use.
pool_size = (3, 3) # Size of pooling area for max pooling.
kernel_size = (3, 3) # Convolutional kernel size.
input_shape = (1, NUM_ROWS, NUM_COLUMNS)

def get_random_move_idx():
    return np.random.choice(4, size=1)[0] 

class RewardPredictor(object):
    def __init__(self):
        # The snake state plane model.
        input_plane = Input(shape=input_shape)

        plane_net = Sequential()
        # 33 * 33 w/ border_mode='same'.
        plane_net.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same', input_shape=input_shape))
        plane_net.add(Activation('relu'))
        # pool_size==stride_size == (3x3). Should get an 11x11 image out.
        plane_net.add(MaxPooling2D(pool_size=pool_size, strides=pool_size))
        plane_net.add(Dropout(0.25))
        plane_net.add(Flatten())
        
        # Encode action.
        action_input = Input(shape=(N_VALID_MOVES,))

        # Encode inputs.
        encoded_plane = plane_net(input_plane)

        # Merge encoded values together.
        merged = merge([action_input, encoded_plane], mode='concat')

        # The reward predicting network.
        final_shape = int( (NUM_ROWS/pool_size[0]) * (NUM_COLUMNS/pool_size[1]) * nb_filters + len(VALID_MOVES) )
        last_stage = Sequential()
        last_stage.add(Dense(256, input_shape=(final_shape,)))
        last_stage.add(Activation('relu'))
        last_stage.add(Dense(32))
        last_stage.add(Activation('relu'))
        last_stage.add(Dense(1))
        last_stage.add(Activation('sigmoid'))

        output = last_stage(merged)

        model = Model(input=[action_input, input_plane], output=output)

        # Compile the model.
        model.compile(loss='mse', optimizer='rmsprop')

        # Save a reference to the model.
        self.model = model

    def fit(self, inputs, targets, **kwargs):
        self.model.fit(inputs, targets, **kwargs)

    def predict(self, inputs):
        return self.model.predict(inputs)

Transition = namedtuple('Transition', ['prev_state', 'action', 'reward', 'new_state', 'terminal'])

class Experience(object):
    def __init__(self, prev_state, action, discounted_reward, new_state):
        self.prev_state = prev_state
        self.action = action
        self.discounted_reward = discounted_reward
        self.new_state = new_state

    def as_image_move_y_tuple(self):
        move_array = np.zeros(4)
        move_array[self.action] = 1
        move = move_array
        image = self.prev_state[np.newaxis,:,:]
        return image, move, self.discounted_reward


def experience_from_many_transitions(transitions):
    t_0 = transitions[0]
    discounted_reward = 0
    for idx, transition in enumerate(transitions):
        # Future rewards are worth a bit less than current rewards.
        discounted_reward += transition.reward * DISCOUNT_FACTOR_GAMMA**idx

    experience = Experience(t_0.prev_state, t_0.action, discounted_reward, t_0.new_state)
    return experience

def experience_from_one_transition(transition):
    return experience_from_many_transitions([transition])

class TransitionSequence(object):
    def __init__(self):
        self.transitions = deque(maxlen=TRANSITION_MEMORY) 
        self.next_transitions = []

    def random_sample(self, sample_size):
        idxs = np.random.choice(len(self.transitions), size=sample_size) 

        experiences = []
        for idx in map(int, idxs):
            game_transitions = list(takewhile(lambda t: not t.terminal, islice(self.transitions, idx, None)))
            num_transitions = len(game_transitions)
            game_transitions.append(self.transitions[idx + num_transitions])
            if len(game_transitions) <= 0:
                raise NotImplementedException()
            if len(game_transitions) == 1:
                experience = experience_from_one_transition(self.transitions[idx])
            else:
                experience = experience_from_many_transitions(game_transitions)
            experience = experience_from_many_transitions(game_transitions)
            experiences.append(experience)

        return experiences

    def record_transition(self, prev_state, action, reward, new_state, terminal):
        self.next_transitions.append(Transition(prev_state, action, reward, new_state, terminal))

    def is_ready_to_train(self):
        return len(self.transitions) > 0 

    def finish_game(self):
        for nt in self.next_transitions:
            self.transitions.append(nt)

def trigger_and_return_action(game, sequence, model):
    p = np.random.random(1)
    if p < RANDOM_MOVE_PROBABILITY:
        move_idx = get_random_move_idx()
    else:
        move_idx = get_model_prediction_idx(game, model)
    game.keypress(VALID_MOVES[move_idx])
    return move_idx

def do_learning(model, sequence):
    experiences = sequence.random_sample(RANDOM_SAMPLE_BATCH_SIZE)
    images = []
    moves = []
    rewards = []
    for experience in experiences:
        image, move, y = experience.as_image_move_y_tuple()
        images.append(image)
        moves.append(move)
        rewards.append(y)

    rewards = np.array(rewards)
    images = np.array(images)
    moves = np.array(moves)

    model.fit([moves, images], rewards, batch_size=RANDOM_SAMPLE_BATCH_SIZE, nb_epoch=1, verbose=0)

def do_game(sequence, model, game_number):
    game = Game(graphical=False)
    while True: # Iterate moves.
        sys.stdout.flush()
        prev_state = game.state.plane.copy()
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
            if game_number % 10 == 0:
                print('food: {} moves: {}'.format(str(e.score.food).zfill(3), e.score.moves))
        new_state = game.state.plane.copy()

        # Save this transition.
        sequence.record_transition(prev_state, action, reward, new_state, terminal)

        # Begin the learning phase
        if sequence.is_ready_to_train():
            do_learning(model, sequence)
        if terminal:
            sequence.finish_game()
            return game.state.moves 


def do_execution(model):
    print('Beginning Execution')
    sequence = TransitionSequence()
    game_number = 0
    moves_history = []
    while True: # Iterate game.
        moves = do_game(sequence, model, game_number)
        moves_history.append(moves)
        game_number += 1
        if game_number % 100 == 0:
            print('Avg Moves = {:0.2f}'.format(np.mean(moves_history)))
            moves_history = []


def main():
    if 'scratch' in sys.argv:
        model = RewardPredictor()
    else:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    try:
        do_execution(model)
    except KeyboardInterrupt:
        print()
        print("KeyboardInterrupt Received. Writing Model.")
        with open('model.pkl', 'wb') as f:
            sys.setrecursionlimit(25000)
            pickle.dump(model, f)

if __name__ == '__main__':
    main()
