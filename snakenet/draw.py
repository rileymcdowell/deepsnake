import pygame

from snakenet.colors import *
from snakenet.game_constants import PIXEL_SIZE, PAD_ROWS, PAD_COLUMNS, RESOLUTION

def _color_cell(pix_array, row, column, color):
    row_start = row*PIXEL_SIZE + PIXEL_SIZE*PAD_ROWS
    row_end = row_start + PIXEL_SIZE
    column_start = column*PIXEL_SIZE + PIXEL_SIZE*PAD_COLUMNS
    column_end = column_start + PIXEL_SIZE 
    pix_array[column_start:column_end, row_start:row_end] = color 

def draw_plane(game):
    game.window_surface.fill(GRAY)

    pix_array = pygame.PixelArray(game.window_surface)

    # Black Board
    pix_array[PIXEL_SIZE:RESOLUTION[1]-PIXEL_SIZE, PIXEL_SIZE:RESOLUTION[0]-PIXEL_SIZE] = BLACK

    # Draw the snake.
    for (row, column) in game.state.snake_deque:
        _color_cell(pix_array, row, column, WHITE)

    # Draw the food 
    _color_cell(pix_array, game.state.food_position[0], game.state.food_position[1], RED)
        
    # Do cleanup
    del pix_array

