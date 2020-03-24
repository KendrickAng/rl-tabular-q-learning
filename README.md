# rl-tabular-q-learning
Entry-level reinforcement learning example with tabular Q-learning. This project follows MorvanZhou's tabular Q-learning tutorials on Github.

Python version: 3.6

## Appendix
* There are two types of reinforcement learning: Utility learning and Q-learning.
* In Utility learning, the agent learns a utility function to maximise utility from any action taken at any state. It thus has to have a model of the environment to be able to evaluate future actions (and hence assign a utility value).
* In Q-learning, the agent learns to assign utilities to each action taken at every state. Hence no model of the environment is required. However, this may restrict the agent's ability to learn.
* When the state space becomes too large, traditional Q-Learning using Q-tables will take too long to converge since it must explore every state-action pair to find the optimal policy.
* In this case, deep reinforcement learning is a viable alternative since the neural network acts as a function approximator to **predict** general Q-values. 
* Deep neural networks therefore allow reinforcement learning to be applied to larger, more complex problems. 
* Any other function approximator can be used in place of a deep neural network.
* "Deep Q-learning" in this context refers to doing Q-learning with a neural network function approximator **together** with the techniques that make it work!
    * Deep Q-learning on its own is unstable and prone to diverging
    * Usually refers to DQN together with Experience Replay and a Separate Target Network.
    