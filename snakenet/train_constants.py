from snakenet.game_constants import VALID_MOVES

TRANSITION_MEMORY = int(1e6) # Number of 'inter-frames' to remember
DISCOUNT_FACTOR_GAMMA = 0.99
N_VALID_MOVES = len(VALID_MOVES)

ALIVE_POINTS = 0.01
EATING_POINTS = 1
USELESS_MOVE_POINTS = -0.1
DYING_POINTS = -1

# Size of a single batch.
SAMPLE_BATCH_SIZE = 32 
# How many minibatches between target network updates?
TARGET_NETWORK_UPDATE_FREQUENCY = 30000
# How many transitions before starting training?
TRANSITION_START_SIZE = 50000

# Filters, filter size, padding.
CONV_CONFIG = [ (64, (5, 5), 'same') 
              #, (32, (6, 6), 'same')
              ]
              
POOL_SIZE = (2, 2) 

# Start at 1.0. Decline linearly for NUM_DECLINING_EPOCHS towards RANDOM_END_P.
NUM_DECLINING_EPOCHS = int(1e6)
RANDOM_END_P = 0.05

RANDOM_MOVE_ACTION_VALUES = (None,)*4

