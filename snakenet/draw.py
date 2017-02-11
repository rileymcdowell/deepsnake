import pygame
import numpy as np

from snakenet.colors import *
from snakenet.game_constants import PIXEL_SIZE, PAD_ROWS, PAD_COLUMNS, RESOLUTION
from snakenet.game_constants import SNAKE_HEAD, SNAKE_VALUE, EMPTY_VALUE, FOOD_VALUE 

TYPE_TO_COLOR = { SNAKE_HEAD: GREEN
                , SNAKE_VALUE: WHITE
                , EMPTY_VALUE: BLACK
                , FOOD_VALUE: RED
                }

def _color_cell(pix_array, row, column, color):
    row_start = row*PIXEL_SIZE + PIXEL_SIZE*PAD_ROWS
    row_end = row_start + PIXEL_SIZE
    column_start = column*PIXEL_SIZE + PIXEL_SIZE*PAD_COLUMNS
    column_end = column_start + PIXEL_SIZE 
    pix_array[column_start:column_end, row_start:row_end] = color 

def draw_plane(game):
    game.window_surface.fill(GRAY)

    pix_array = pygame.PixelArray(game.window_surface)
    pixel_iterator = np.nditer(game.state.plane, flags=['multi_index'])

    while not pixel_iterator.finished: 
        row, column = tuple(pixel_iterator.multi_index)
        value = pixel_iterator[0]
        color = TYPE_TO_COLOR[int(value)]
        _color_cell(pix_array, row, column, color)
        pixel_iterator.iternext()

    # Do cleanup
    del pix_array

