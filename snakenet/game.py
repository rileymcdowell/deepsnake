import pygame

from snakenet.game_constants import SCREEN_SIZE, RIGHT, LEFT, UP, DOWN
from snakenet.game_state import GameState

FLAGS = 0 # No flags
COLOR_DEPTH = 32 # bits
CAPTION = "Snake!"

class Game(object):
    def __init__(self, graphical=True):
        if graphical:
            self.window_surface = pygame.display.set_mode(SCREEN_SIZE, FLAGS, COLOR_DEPTH)
            pygame.display.set_caption(CAPTION)
        self.state = GameState()

    def keypress(self, direction):
        """ 
        Allow movements, but not in the exact opposite direction of the current movement.
        Return true for allowed movements, false if the keypress indicated opposite
        direction movement.
        """
        if direction == UP and self.state.last_moved != DOWN :
            self.state.last_pressed = UP
            return True
        elif direction == DOWN and self.state.last_moved != UP:
            self.state.last_pressed = DOWN
            return True
        elif direction == RIGHT and self.state.last_moved != LEFT:
            self.state.last_pressed = RIGHT
            return True
        elif direction == LEFT and self.state.last_moved != RIGHT:
            self.state.last_pressed = LEFT
            return True
        else:
            # Indicator of a useless move - exact opposite of current direction.
            # In this case, we continue in the last good keypress direction anyway.
            self.state.last_pressed = self.state.last_moved
            return False

    def move(self, action_values=None):
        self.state.process_move(action_values)

