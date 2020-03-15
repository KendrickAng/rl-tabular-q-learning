# rl-tabular-q-learning
Entry-level reinforcement learning example with tabular Q-learning. This project follows MorvanZhou's tabular Q-learning tutorials on Github.

## Appendix
* There are two types of reinforcement learning: Utility learning and Q-learning.
* In Utility learning, the agent learns a utility function to maximise utility from any action taken at any state. It thus has to have a model of the environment to be able to evaluate future actions (and hence assign a utility value).
* In Q-learning, the agent learns to assign utilities to each action taken at every state. Hence no model of the environment is required. However, this may restrict the agent's ability to learn.