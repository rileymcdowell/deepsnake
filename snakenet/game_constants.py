import pygame

from collections import namedtuple

# Screen Constants
PIXEL_SIZE = 16 
NUM_ROWS = 16 + 1 
NUM_COLUMNS = 16 + 1 
PAD_ROWS = 1
PAD_COLUMNS = 1
RESOLUTION = (PIXEL_SIZE * (NUM_ROWS+PAD_ROWS*2), PIXEL_SIZE * (NUM_COLUMNS+PAD_COLUMNS*2))

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

