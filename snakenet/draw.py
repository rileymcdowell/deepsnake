import pygame
import numpy as np

from itertools import permutations
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from snakenet.colors import GREEN, WHITE, BLACK, RED, GRAY
from snakenet.game_constants import PIXEL_SIZE, PAD_ROWS, PAD_COLUMNS, RESOLUTION
from snakenet.game_constants import SNAKE_HEAD, SNAKE_VALUE, EMPTY_VALUE, FOOD_VALUE 
from snakenet.game_constants import NUM_ROWS, NUM_COLUMNS, LEFT_BAR_WIDTH 
from snakenet.game_constants import ZOOM_GRID_ROW_OFFSET, ACTION_VALUE_ROW_OFFSET


TYPE_TO_COLOR = { SNAKE_HEAD: GREEN
                , SNAKE_VALUE: WHITE
                , EMPTY_VALUE: BLACK
                , FOOD_VALUE: RED
                }

def _color_board_cell(pix_array, row, column, color):
    row_start = row*PIXEL_SIZE + PIXEL_SIZE*PAD_ROWS
    row_end = row_start + PIXEL_SIZE
    column_start = column*PIXEL_SIZE + PIXEL_SIZE*PAD_COLUMNS + LEFT_BAR_WIDTH
    column_end = column_start + PIXEL_SIZE 
    pix_array[column_start:column_end, row_start:row_end] = color

def _color_zoom_cell(pix_array, row, column, color):
    row_start = row*PIXEL_SIZE + PIXEL_SIZE*PAD_ROWS + ZOOM_GRID_ROW_OFFSET * PIXEL_SIZE
    row_end = row_start + PIXEL_SIZE
    column_start = column*PIXEL_SIZE + PIXEL_SIZE*PAD_COLUMNS
    column_end = column_start + PIXEL_SIZE 
    pix_array[column_start:column_end, row_start:row_end] = color

def _color_action_value_cell(pix_array, row, column, color):
    row_start = row*PIXEL_SIZE + PIXEL_SIZE*PAD_ROWS + ACTION_VALUE_ROW_OFFSET * PIXEL_SIZE
    row_end = row_start + PIXEL_SIZE
    column_start = column*PIXEL_SIZE + PIXEL_SIZE*PAD_COLUMNS
    column_end = column_start + PIXEL_SIZE 
    pix_array[column_start:column_end, row_start:row_end] = color

def _draw_colorbar(pix_array, colormap):
    row_start = ACTION_VALUE_ROW_OFFSET * PIXEL_SIZE + 5 * PIXEL_SIZE
    row_end = row_start + PIXEL_SIZE
    column_start = PIXEL_SIZE
    column_stop = column_start + 2*PIXEL_SIZE
    steps = np.arange(10) 
    offset = (column_stop - column_start) / len(steps)
    colors = _get_colors(steps, colormap)
    for color, step in zip(colors, steps):
        col_s = int(column_start + offset*step)
        col_e = int(column_stop - offset*step)
        pix_array[col_s:col_e, row_start:row_end] = color

def _get_colors(array, cmap):
    cmap.set_array(array)
    cmap.autoscale()
    colors = cmap.to_rgba(array) * 255
    colors = colors[...,:3] # Remove alpha channel.
    colors = colors.astype(np.uint8) # Convert to uint8.
    colors = list(map(tuple, colors)) # pygame wants a tuple of 3 ints. 
    return colors

def draw_game(game):
    """
    Ugly function that draws the game board by writing to an array of pixels.
    """
    game.window_surface.fill(GRAY)

    pix_array = pygame.PixelArray(game.window_surface)

    # Draw the complete board.
    pixel_iterator = np.nditer(game.state.plane, flags=['multi_index'])
    while not pixel_iterator.finished: 
        row, column = tuple(pixel_iterator.multi_index)
        value = pixel_iterator[0]
        color = TYPE_TO_COLOR[int(value)]
        _color_board_cell(pix_array, row, column, color)
        pixel_iterator.iternext()

    # Draw the zoomed board.
    head_row, head_col = game.state.snake_position
    rows = (head_row - 1, head_row, head_row + 1)
    cols = (head_col - 1, head_col, head_col + 1)
    for row_idx, row in enumerate(rows):
        for col_idx, col in enumerate(cols):
            # Ugly, but we don't want to try to grab an index off the plane. 
            good_row = row >= 0 and row < NUM_ROWS
            good_col = col >= 0 and col < NUM_COLUMNS
            if good_row and good_col:
                color = TYPE_TO_COLOR[int(game.state.plane[row, col])]
            else:
                color = GRAY
            _color_zoom_cell(pix_array, row_idx, col_idx, color)

    # Draw the action value colormap.
    # This is not visible in the case of a human player. 
    up = (0, 1)
    down = (2, 1)
    left = (1, 0)
    right = (1, 2)
    cm = ScalarMappable(cmap='plasma')
    action_values = np.array(game.state.action_values)
    if not np.equal(action_values, (None,)*4).all():
        colors = _get_colors(action_values, cm)
    else:
        # All None values indicates that the model didn't evaluate this move.
        colors = (GRAY,)*4
    for (row, col), color in zip([up, down, left, right], colors):
        _color_action_value_cell(pix_array, row, col, color)

    #_draw_colorbar(pix_array, cm)
     

    # Do cleanup
    del pix_array

    pygame.display.update()

