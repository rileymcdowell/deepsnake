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

def get_model_prediction_idx(game, model):
    data = game.state.plane[np.newaxis,:,:][np.newaxis,...]
    predicted_rewards = []
    for move_idx, move in enumerate(VALID_MOVES):
        move_array = np.zeros(4)
        move_array[move_idx] = 1
        move_array = move_array[np.newaxis,:]
        model_output = model.predict([move_array, data]) # Predict one item.
        predicted_rewards.append(model_output[0])
    max_prediction_idx = np.argmax(predicted_rewards)
    return max_prediction_idx 

def get_model_keypress(game):
    model = get_model()
    predicted_best_move_idx = get_model_prediction_idx(game, model) 
    predicted_best_move = VALID_MOVES[predicted_best_move_idx]
    return MOVE_TO_KEYPRESS[predicted_best_move]
