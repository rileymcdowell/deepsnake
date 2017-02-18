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

def get_future_rewards(image, model):
    image = image[np.newaxis,:,:][np.newaxis,...]
    moves = []
    for move_idx, move in enumerate(VALID_MOVES):
        move_array = np.zeros(4)
        move_array[move_idx] = 1
        moves.append(move_array)

    move_input = np.array(moves)
    image_input = np.repeat(image, 4, axis=0)

    predicted_rewards = model.predict([move_input, image_input])

    return predicted_rewards

def get_model_prediction_idx(game, model):
    """
    This is the equivalent of the policy (pi). It converts the
    output of the Q-function to an action. In this case, the
    action is selecting the index of a move to perform.
    """
    image = game.state.plane
    predicted_rewards = get_future_rewards(image, model) 
    max_prediction_idx = np.argmax(predicted_rewards)
    return max_prediction_idx 

def get_model_keypress(game):
    model = get_model()
    predicted_best_move_idx = get_model_prediction_idx(game, model) 
    predicted_best_move = VALID_MOVES[predicted_best_move_idx]
    return MOVE_TO_KEYPRESS[predicted_best_move]
