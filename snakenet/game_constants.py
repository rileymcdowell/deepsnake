from collections import namedtuple

# Screen Constants
PIXEL_SIZE = 8
NUM_ROWS = 32 + 1 
NUM_COLUMNS = 32 + 1 
PAD_ROWS = 1
PAD_COLUMNS = 1
RESOLUTION = (PIXEL_SIZE * (NUM_ROWS+PAD_ROWS*2), PIXEL_SIZE * (NUM_COLUMNS+PAD_COLUMNS*2))

# Directions 
DOWN = 'd'
UP = 'u'
RIGHT = 'r'
LEFT = 'l'

# Snake Constants
SNAKE_INITIALSIZE = 4
SNAKE_GROWBY = 3

# Plane Constants
SNAKE_VALUE = 0 
EMPTY_VALUE = 1 
ITEM_VALUE = 2

Score = namedtuple('Score', ['food', 'moves'])

# Exceptional States
class LoseException(Exception):
    def __init__(self, food, moves):
        self.score = Score(food, moves)
        msg = "You Lose. Ate {} times. Moved {} times.".format(self.score.food, self.score.moves)
        super(LoseException, self).__init__(msg)

