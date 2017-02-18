
import sys
import time
import pygame
import pygame.locals as pl

from argparse import ArgumentParser

from snakenet.draw import draw_plane
from snakenet.game import Game
from snakenet.game_constants import DOWN, UP, RIGHT, LEFT
from snakenet.model_player import get_model_keypress, warmup_model_player
from snakenet.train import QNetwork 

QNetwork = QNetwork

QUIT = 'quit'
TICK = pygame.USEREVENT + 1

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input', choices=['user', 'model'], default='user')
    return parser.parse_args()

ARGS = parse_args()

def process_tick(game):
    game.move()

def process_keypress(game, key):
    if key == pygame.K_UP:
        game.keypress(UP)
    if key == pygame.K_DOWN:
        game.keypress(DOWN)
    if key == pygame.K_RIGHT:
        game.keypress(RIGHT)
    if key == pygame.K_LEFT:
        game.keypress(LEFT)

def process_event(game, event):
    if event.type == pl.QUIT:
        sys.exit(0)
    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        sys.exit(0)

    if event.type == pygame.KEYDOWN:
        if ARGS.input == 'user':
            process_keypress(game, event.key)

    if event.type == TICK:
        if ARGS.input == 'model':
            key = get_model_keypress(game)
            process_keypress(game, key)
        # Update the game.
        process_tick(game)
        # Refresh the image.
        draw_plane(game)
        pygame.display.update()

def process_events(game, clock):
    for event in pygame.event.get():
        process_event(game, event)

def mainloop(game):
    clock = pygame.time.Clock()
    pygame.time.set_timer(TICK, 100)
    while True:
        process_events(game, clock)
        clock.tick(60)

def do_warmup():
    print('Warming up model player...', end=' ')
    warmup_model_player()
    print('Warmup completed.')

def main():
    if ARGS.input == 'model':
        # Don't let the game initialize before the model player is ready.
        do_warmup()

    pygame.init()
    game = Game()

    # Do an initial draw.
    draw_plane(game)
    pygame.display.update()

    # Wait for everything to complete.
    time.sleep(0.2)

    # Enter the mainloop.
    try:
        mainloop(game)
    except (KeyboardInterrupt, SystemExit):
        pygame.quit()
        sys.exit()
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == '__main__':
    main()
