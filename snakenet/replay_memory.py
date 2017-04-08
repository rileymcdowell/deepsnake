import numpy as np
from heapq import heappush, heappushpop, heappop, nlargest
from snakenet.train_constants import TRANSITION_START_SIZE, TRANSITION_MEMORY
from collections import namedtuple

fields = ['td_error', 'old_state', 'action', 'reward', 'new_state', 'terminal']
Transition = namedtuple('Transition', fields)

#SAMPLE_METHOD = 'greedy'
SAMPLE_METHOD = 'random'

class Transition(object):
    def __init__(self, td_error, old_state, action, reward, new_state, terminal):
        self.td_error = td_error
        self.old_state = old_state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.terminal = terminal

    def __gt__(self, other):
        return self.td_error > other.td_error 

    def __lt__(self, other):
        return self.td_error < other.td_error 

    def __eq__(self, other):
            return self.td_error == other.td_error

class ReplayMemory(object):
    def __init__(self):
        self.transitions = [] 
        self.next_transitions = []

    def sample(self, sample_size):
        """ Sample from replay memory """
        samples = []
        if SAMPLE_METHOD == 'greedy':
            for _ in range(sample_size):
                samples.append(heappop(self.transitions))
        elif SAMPLE_METHOD == 'random':
            idxs = np.random.randint(0, len(self.transitions), size=sample_size)
            for idx in idxs:
                samples.append(self.transitions[idx])
        return samples

    def _push_self(self, transition):
        """ Implement a maximum heap size """
        # TODO: Implement random selection probability.
        if len(self.transitions) < TRANSITION_MEMORY:
            # We're below capacity, so continue pushing.
            heappush(self.transitions, transition)
        elif len(self.transitions) == TRANSITION_MEMORY:
            # We're at capacity, so discard the popped (smallest) value.
            heappushpop(self.transitions, transition)

    def record_transition(self, old_state, action, reward, new_state, terminal):
        """ Record a new transition that should be considered at maximum training priority. """
        # New transitions get infinite error so they are sure to be visited right away.
        self.next_transitions.append(Transition(float('inf'), old_state, action, reward, new_state, terminal))

    def update_transitions(self, samples, td_errors):
        """ Put a transition back, but with a new, updated td_error. """
        if SAMPLE_METHOD == 'greedy':
            for sample, td_error in zip(samples, td_errors):
                sample.td_error = td_error
                self._push_self(sample)
        elif SAMPLE_METHOD == 'random':
            pass # Random sampling doesn't remove entries.

    def is_ready_to_train(self):
        """ Wait a while before starting training. Use this function to check """
        return len(self.transitions) >= TRANSITION_START_SIZE

    def finish_game(self):
        """ Wait until a game is finished to add the new transitions to replay memory """
        for nt in self.next_transitions:
            self._push_self(nt)


