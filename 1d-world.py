"""
An agent starts on the leftmost side of a 1-dimensional world, and has to find the goal on the rightmost side.
"""

import numpy as np
import pandas as pd
import logging

logging.disabled = False
logging.basicConfig(level=logging.DEBUG)

# Q-learning variables
NUM_STATES = 6      # x-length of the 1d world
ACTIONS = ["LEFT", "RIGHT"] # available actions to agent
EPSILON = 0.2       # percent of time exploring (instead of exploiting)
ALPHA = 0.1         # learning rate
GAMMA = 0.9         # discount factor
MAX_EPISODES = 13   # maximum iterations to update q-table
REFRESH_TIME = 0.3  # refresh time for one move

"""Initialises and returns the empty (NUM_STATES x ACTIONS) q-table."""
def init_q_table():
    q_table = pd.DataFrame(
        data=np.zeros((NUM_STATES, len(ACTIONS))),
        columns=ACTIONS,
    )
    logging.debug("Q-table is\n{0}".format(q_table))
    return q_table

def rl():
    q_table = init_q_table()
    return q_table

def main():
    q_table = rl()

if __name__ == "__main__":
    main()