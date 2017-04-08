import pygame

import numpy as np

from collections import namedtuple


# Board Constants
PIXEL_SIZE = 20 
NUM_ROWS = 11 + 1 
NUM_COLUMNS = 11 + 1 
PAD_ROWS = 1
PAD_COLUMNS = 1
RESOLUTION = ( PIXEL_SIZE * (NUM_ROWS+PAD_ROWS*2), PIXEL_SIZE * (NUM_COLUMNS+PAD_COLUMNS*2) )

LEFT_BAR_WIDTH = PIXEL_SIZE * 4
SCREEN_SIZE = (RESOLUTION[0] + LEFT_BAR_WIDTH, RESOLUTION[1])
ZOOM_GRID_ROW_OFFSET = 0 # Start at top (0) of left bar.
ACTION_VALUE_ROW_OFFSET = NUM_ROWS // 2 # Middle of left bar.


# Directions 
DOWN = 'd'
UP = 'u'
RIGHT = 'r'
LEFT = 'l'
MOVE_TO_KEYPRESS = { UP: pygame.K_UP
                   , DOWN: pygame.K_DOWN
                   , RIGHT: pygame.K_RIGHT
                   , LEFT: pygame.K_LEFT
                   }

# Used by model player and trainer to know what is possible.
VALID_MOVES = np.array([UP, DOWN, LEFT, RIGHT])

# Snake Constants
SNAKE_INITIALSIZE = 4
SNAKE_GROWBY = 3

# Plane Constants, Must Be > 0.
SNAKE_HEAD = 1
SNAKE_VALUE = 2 
EMPTY_VALUE = 3 
FOOD_VALUE = 4 

Score = namedtuple('Score', ['food', 'moves'])

# Exceptional States
class LoseException(Exception):
    def __init__(self, food, moves):
        self.score = Score(food, moves)
        msg = "You Lose. Ate {} times. Moved {} times.".format(self.score.food, self.score.moves)
        super(LoseException, self).__init__(msg)


