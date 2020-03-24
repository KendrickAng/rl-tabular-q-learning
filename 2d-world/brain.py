import numpy as np
import pandas as pd

ALPHA = 0.01    # learning rate
GAMMA = 0.9     # reward decay
EPSILON = 0.9   # probability of exploiting instead of exploring

class QLearningTable:
    def __init__(self, actions, learning_rate=ALPHA, reward_decay=GAMMA, e_greedy=EPSILON):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=actions, dtype=np.float64)

    def select_next_action(self, s_curr):
        self.check_state_exists(s_curr)
        if np.random.uniform() < self.epsilon:
            # exploit
            state_action = self.q_table.loc[s_curr, :]
            # .index decomposes the series into the max actions (the columns).
            return np.random.choice(state_action[state_action == np.max(state_action)].index)
        # explore
        return np.random.choice(self.actions)

    def learn(self, s_curr, action, reward, s_next):
        self.check_state_exists(s_next)
        q_predict = self.q_table.loc[s_curr, action]
        q_target = reward + self.gamma * self.q_table.loc[s_next, :].max()  # SARSA
        if s_next == "TERMINAL":
            q_target = reward
        self.q_table.loc[s_curr, action] += self.lr * (q_target - q_predict)

    """Adds the state to the q-table, if it isn't already inside."""
    def check_state_exists(self, state):
        if state not in self.q_table.index:
            # append new state to the q table
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state)
            )