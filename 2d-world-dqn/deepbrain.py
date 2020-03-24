"""
Deep Q-Network replaces the Q-table in the previous 2d world.

Tensorflow: 1.2
"""
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:
    def __init__(
            self, n_actions, n_features, learning_rate=0.01,
            reward_decay=0.9, e_greedy=0.9, replace_target_iter=300,
            memory_size=500, batch_size=32, e_greedy_increment=None, output_graph=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning steps used so far
        self.learn_step_counter = 0

        # initialise the empty memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consists [target_net, evaluate_net]
        self._build_net()

        # replace parameters in the network
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # input nodes
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s') # input state
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_') # next state
        self.r = tf.placeholder(tf.float32, [None, ], name='r') # input reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a') # input action

        w_initialiser, b_initialiser = tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)

        # our main network that we calculate changes to q-values from, after taking target q-value from the target network.
        with tf.variable_scope('eval_net'):
            # Initialisers simply define the way to set the weights of layers
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initialiser,
                                 bias_initializer=b_initialiser, name='e1')
            # q_eval layer receives input from e1, outputs an action (hence size n_actions)
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initialiser,
                                          bias_initializer=b_initialiser, name='q')

        # evaluation net - calculates target q value only. Its weights are frozen, and periodically copied from the evaluation network.
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initialiser,
                                 bias_initializer=b_initialiser, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initialiser,
                                          bias_initializer=b_initialiser, name='t2')

        with tf.variable_scope('q_target'):
            # reduce_max computes the maximum of elements along the y-axis (max col)
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            self.q_target = tf.stop_gradient(q_target) # TODO: ?
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1) # TODO: ?
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices) # some form of manipulating tensor shape and their variables
        with tf.variable_scope('loss'):
            # mean squared error loss
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    # Create memory for experience replay
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # squeeze into a row
        transition = np.hstack((s, [a, r], s_))
        # replace old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    # input the current (s,a,r,s_) from memory and get an output
    def learn(self):
        # replace target network params from eval (live) network every so often
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print("\ntarget_params_replaced\n")

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            }
        )
        # for logging purposes
        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()