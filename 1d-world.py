"""
An agent starts on the leftmost side of a 1-dimensional world, and has to find the goal on the rightmost side.
"""
import time

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()
log.disabled = True

# Q-learning variables
NUM_STATES = 6      # x-length of the 1d world
TERMINAL_STATE = NUM_STATES - 1 # terminal state is rightmost state
ACTIONS = ["LEFT", "RIGHT"] # available actions to agent
EPSILON = 0.9       # percent of time exploiting (instead of exploring)
ALPHA = 0.1         # learning rate
GAMMA = 0.9         # discount factor
MAX_EPISODES = 15   # An episode doesn't end until the agent reaches the goal
REFRESH_TIME = 0.25  # refresh time for one move

"""Initialises and returns the empty (NUM_STATES x ACTIONS) q-table."""
def init_q_table(num_states, actions):
    q_table = pd.DataFrame(
        data=np.zeros((num_states, len(actions))),
        columns=ACTIONS,
    )
    log.debug("\rQ-table is\n{0}".format(q_table))
    return q_table

"""Based current state, select the best action with highest utility in the q-table."""
def select_next_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or (state_actions == 0).all(): # initially, when all actions have 0 episolon
        action = np.random.choice(ACTIONS)
    else:
        action = state_actions.idxmax()
    log.debug("\rPicked {0}".format(action))
    return action

"""Returns the next state and reward from this state based on action taken at current state."""
def get_env_feedback(state, action):
    if action == "RIGHT":
        if state == TERMINAL_STATE - 1:
            return TERMINAL_STATE, 1
        return state + 1, 0
    if action == "LEFT":
        if state == 0:
            return state, 0         # bump into wall
        return state - 1, 0
    raise RuntimeError()

"""Prints the current environment, taking into account agent's state, current episode and step counter."""
def display_env(state, episode, step_counter):
    # This is how environment be updated
    env_list = ['-'] * (NUM_STATES - 1) + ['T']  # '---------T' our environment
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(REFRESH_TIME)

"""Q-learning using the SARSA (State-Action-Reward-State-Action) update rule"""
def rl():
    q_table = init_q_table(NUM_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        s_curr = 0           # agent starts at leftmost position
        step_counter = 0
        at_goal = False
        while not at_goal:
            display_env(s_curr, episode, step_counter)

            action = select_next_action(s_curr, q_table)
            s_next, reward = get_env_feedback(s_curr, action)
            q_predict = q_table.loc[s_curr, action]
            q_target = reward + GAMMA * q_table.iloc[s_next, :].max()  # max q-value from next state

            if s_next == TERMINAL_STATE:
                # value iteration for q-learning: Q(st, at) = r(st, at) + GAMMA * max q-value from any action from st
                q_target = reward
                at_goal = True

            # q-value update
            q_table.loc[s_curr, action] += ALPHA * (q_target - q_predict)
            s_curr = s_next
            step_counter += 1
        log.debug("\rCompleted episode {0}".format(episode))
        log.debug("\r{}".format(q_table))

    return q_table

def main():
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

if __name__ == "__main__":
    main()