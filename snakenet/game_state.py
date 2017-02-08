import numpy as np

from snakenet.game_constants import NUM_ROWS, NUM_COLUMNS, SNAKE_INITIALSIZE, SNAKE_VALUE, EMPTY_VALUE, LoseException, SNAKE_GROWBY 
from snakenet.game_constants import UP, DOWN, RIGHT, LEFT
from collections import deque

class GameState(object):
    def __init__(self):
        self.plane = np.full((NUM_ROWS, NUM_COLUMNS), fill_value=EMPTY_VALUE, dtype=np.uint8)
        self.snake_deque = deque(maxlen=SNAKE_INITIALSIZE)
        self.snake_position = None
        self.food_position = None
        self.last_pressed = None

        # Setup the initial game state.
        self.initialize()
        self.set_random_food_position()

        # Track score
        self.times_eaten = 0
        self.moves = 0

    def initialize(self):
        middle_row = NUM_ROWS // 2
        middle_column = NUM_COLUMNS // 2

        for i in range(SNAKE_INITIALSIZE):
            # End the last loop exactly in the middle.
            row = middle_row + SNAKE_INITIALSIZE - i - 1
            self.set_snake(row, middle_column) 

    def set_random_food_position(self):
        while True:
            row = np.random.choice(np.arange(0, NUM_ROWS, dtype=np.uint8)) 
            column = np.random.choice(np.arange(0, NUM_COLUMNS, dtype=np.uint8))
            if (row, column) in self.snake_deque:
                continue
            else:
                self.food_position = (row, column)
                break

    def eat_food(self):
        self.snake_deque = deque(self.snake_deque, maxlen=self.snake_deque.maxlen + SNAKE_GROWBY)
        self.set_random_food_position()
        self.times_eaten += 1

    def lose(self):
        raise LoseException(self.times_eaten, self.moves)

    def set_snake(self, row, column):
        position = (row, column)
        if row < 0 or row >= NUM_ROWS:
            self.lose()
        if column < 0 or column >= NUM_COLUMNS:
            self.lose()
        if position in self.snake_deque:
            self.lose()

        self.plane[row, column] = SNAKE_VALUE
        self.snake_deque.appendleft((row, column))
        self.snake_position = (row, column)

        if position == self.food_position:
            self.eat_food()

    @property
    def row(self):
        return self.snake_position[0] 

    @property
    def column(self):
        return self.snake_position[1]

    def process_move(self):
        if self.last_pressed == UP:
            self.set_snake(self.row - 1, self.column)
        if self.last_pressed == DOWN:
            self.set_snake(self.row + 1, self.column)
        if self.last_pressed == RIGHT:
            self.set_snake(self.row, self.column + 1)
        if self.last_pressed == LEFT:
            self.set_snake(self.row, self.column - 1)
        self.moves += 1
            
