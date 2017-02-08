import pickle
import numpy as np

from uuid import uuid4
from collections import namedtuple, deque 
from keras.models import Sequential
from keras.layers import Dense, Activation

from snakenet.game import Game
from snakenet.game_constants import DOWN, UP, LEFT, RIGHT, NUM_ROWS, NUM_COLUMNS, LoseException

VALID_MOVES = np.array([UP, DOWN, LEFT, RIGHT])

TRANSITION_MEMORY = 1000 # Number of 'inter-frames' to remember
DISCOUNT_FACTOR = 0.01
RANDOM_MOVE_PROBABILITY = 0.05

ALIVE_POINTS = 1
EATING_POINTS = 1000
DYING_POINTS = -1000

def get_random_move():
    return np.random.choice(4, size=1) 

def get_nn():
    model = Sequential()
    model.add(Dense(output_dim=20, input_dim=NUM_ROWS*NUM_COLUMNS))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=4))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model # ready for training and execution.

def get_model_prediction(game, model):
    data = game.state.plane.ravel()
    res = model.predict(data)
    return res

Transition = namedtuple('Transition', ['prev_state', 'action', 'reward', 'new_state', 'terminal'])
Experience = namedtuple('Experience', ['prev_state', 'action', 'total_reward', 'new_state'])

class ExperienceSequence(object):
    def __init__(self):
        self.transitions = deque(maxlen=TRANSITION_MEMORY) 

    def random_sample(self):

    def add_new_experience(self, prev_state, action, reward, new_state, terminal):
        self.transitions.append(Transition(prev_state, action, reward, new_state, terminal))

def perform_action(game, sequence, model):
    p = np.random.random(1)  
    if p < RANDOM_MOVE_PROBABILITY:
        move = get_random_move()
    else:
        move = get_model_prediction(game, model)
    game.keypress(move)
    return move

def do_learning(model):
    sequence = ExperienceSequence()
    while True: # Iterate game.
        game = Game() 
        while True: # Iterate moves.
            prev_state = game.state.plane
            old_times_eaten = game.state.times_eaten
            # Guess that we don't die.
            terminal = False
            reward = ALIVE_POINTS 
            try:
                action = perform_action(game, sequence, model)
                new_times_eaten = game.state.times_eaten
                if new_times_eaten > old_times_eaten: 
                    reward = EATING_POINTS # Eating is really good.
            except LoseException as e:
                reward = DYING_POINTS # Game over is really bad.
                terminal = True
            new_state = game.state.plane

            # Save this experience.
            sequence.add_new_experience(prev_state, action, reward, new_state)

def main():
    model = get_nn()
    try:
        do_learning(model)
    except:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

if __name__ == '__main__':
    main()
