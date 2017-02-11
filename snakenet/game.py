import pygame

from snakenet.game_constants import RESOLUTION, RIGHT, LEFT, UP, DOWN
from snakenet.game_state import GameState

FLAGS = 0 # No flags
COLOR_DEPTH = 32 # bits
CAPTION = "Snake!"

class Game(object):
    def __init__(self, graphical=True):
        if graphical:
            self.window_surface = pygame.display.set_mode(RESOLUTION, FLAGS, COLOR_DEPTH)
            pygame.display.set_caption(CAPTION)
        self.state = GameState()

    def keypress(self, direction):
        """ Allow movements, but not in the exact opposite direction """
        if direction == UP and self.state.last_moved != DOWN :
            self.state.last_pressed = UP
        elif direction == DOWN and self.state.last_moved != UP:
            self.state.last_pressed = DOWN
        elif direction == RIGHT and self.state.last_moved != LEFT:
            self.state.last_pressed = RIGHT
        elif direction == LEFT and self.state.last_moved != RIGHT:
            self.state.last_pressed = LEFT

    def move(self):
        self.state.process_move()


