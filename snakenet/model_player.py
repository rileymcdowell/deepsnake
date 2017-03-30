import pygame
import pickle
import numpy as np

from snakenet.game_constants import DOWN, UP, LEFT, RIGHT, MOVE_TO_KEYPRESS

VALID_MOVES = np.array([UP, DOWN, LEFT, RIGHT])

_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None:
        with open('model.pkl', 'rb') as f:
            _MODEL = pickle.load(f)
    return _MODEL

def warmup_model_player():
    get_model() 

def get_action_values(image, model):
    image = image[np.newaxis,...] # Add color channel.
    image = image[np.newaxis,...] # Make it a singleton list.
    action_values = model.predict([image])[0] # Extract the values.
    return action_values

def get_model_prediction_idx(game, model):
    """
    This is the equivalent of the policy (pi). It converts the
    output of the Q-function to an action. In this case, the
    action is selecting the index of a move to perform.
    """
    image = game.state.plane
    action_values = get_action_values(image, model)
    max_prediction_idx = np.argmax(action_values)
    return max_prediction_idx, action_values 

def get_model_keypress(game):
    model = get_model()
    predicted_best_move_idx, action_values = get_model_prediction_idx(game, model) 
    predicted_best_move = VALID_MOVES[predicted_best_move_idx]
    return MOVE_TO_KEYPRESS[predicted_best_move], action_values
