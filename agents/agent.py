import numpy as np
from tasks.task import Task
import random
from collections import namedtuple, deque
import tensorflow as tf

from memory import RingBuffer as ReplayBuffer

# from keras import layers, models, optimizers
# from keras import backend as K

from tensorflow.contrib.keras import layers, models, optimizers
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import activations
from tensorflow.contrib.keras import regularizers
from tensorflow.contrib.keras import initializers

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, gym=False):
        self.task = task
        if gym:
            self.state_size = np.prod(task.observation_space.shape)
            self.action_size = np.prod(task.action_space.shape)
            self.action_low = task.action_space.low
            self.action_high = task.action_space.high
        else:
            self.state_size = task.state_size
            self.action_size = task.action_size
            self.action_low = task.action_low
            self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        # self.exploration_mu = 0.0
        # self.exploration_theta = 0.15
        # self.exploration_sigma = 0.2
        # self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 1e-2  # for soft update of target parameters

        # Score
        self.score = 0.0

    def reset_episode(self):
        # self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.score = 0.0
        # self.noise.sigma = max(0.001, self.noise.sigma*0.99)
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

        # Add in step reward to episode score
        self.score += reward

    def add_to_memory(self, last_state, action, reward, next_state, done):
        self.memory.add(last_state, action, reward, next_state, done)

    # Action without noise
    # def act(self, states):
    #     """Returns actions for given state(s) as per current policy."""
    #     state = np.reshape(states, [-1, self.state_size])
    #     action = self.actor_local.model.predict(state)[0]
    #     return action

    # Action with noise
    def act(self, states):
        """Returns actions for given state(s) as per current policy with added noise for exploration."""
        # normalize state
        if hasattr(self.memory, 'state_norm') and self.memory.state_norm is not None:
            states = self.memory.state_norm.normalize(states)

        state = np.reshape(states, [-1, self.state_size])

        action = self.actor_local.model.predict(state)[0]
        # add some noise for exploration
        # noise = self.noise.sample()#*0.5*(self.action_high - self.action_low)
        # action = np.clip(action + noise, self.action_low, self.action_high)
        return action

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        # states = np.vstack([e.state for e in experiences if e is not None])
        # actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        # rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        # dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        # next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Unpack experiences
        states, actions, rewards, next_states, dones = experiences

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)

        # Train critic model (local)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]),
                                      (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        kernel_l2_reg = 1e-5

        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # size_repeat = 20
        # block_size = size_repeat*self.state_size
        # print("Actor block size = {}".format(block_size))

        # net = layers.concatenate([states]*size_repeat)
        # for _ in range(5):
        #     net = res_block(net, block_size)

        # Add hidden layers
        # states = layers.BatchNormalization()(states)
        net = layers.Dense(units=300,
                           # activation='relu',
                           activation=None,
                           # kernel_initializer='lecun_normal',
                           # activity_regularizer=regularizers.l2(0.1),
                           kernel_regularizer=regularizers.l2(kernel_l2_reg),
                           # bias_initializer=initializers.Constant(1e-2),
                           use_bias=True)(states)
        # net = layers.BatchNormalization()(net)
        net = layers.LeakyReLU(1e-2)(net)
        # net = layers.ELU()(net)

        net = layers.Dense(units=400,
                           # activation='relu',
                           activation=None,
                           # kernel_initializer='lecun_normal',
                           # activity_regularizer=regularizers.l2(0.01),
                           kernel_regularizer=regularizers.l2(kernel_l2_reg),
                           # bias_initializer=initializers.Constant(1e-2),
                           use_bias=True)(net)
        # net = layers.BatchNormalization()(net)
        net = layers.LeakyReLU(1e-2)(net)
        # net = layers.ELU()(net)
        #
        net = layers.Dense(units=200,
                           # activation='relu',
                           activation=None,
                           # activity_regularizer=regularizers.l2(0.01),
                           kernel_regularizer=regularizers.l2(kernel_l2_reg),
                           # bias_initializer=initializers.Constant(1e-2),
                           use_bias=True)(net)
        # net = layers.BatchNormalization()(net)
        net = layers.LeakyReLU(1e-2)(net)
        # net = layers.ELU()(net)

        # net = layers.Dense(units=32,
        #                    activation='relu',
                           # activation=None,
                           # activity_regularizer=regularizers.l2(0.01),
                           # kernel_regularizer=regularizers.l2(kernel_l2_reg),
                           # use_bias=True)(net)
        # net = layers.BatchNormalization()(net)
        # net = layers.LeakyReLU()(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size,
                                   activation='sigmoid',
                                   kernel_regularizer=regularizers.l2(kernel_l2_reg),
                                   kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                   # bias_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                   name='raw_actions')(net)
        #
        # # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)

        # actions = layers.Dense(units=self.action_size,
        #                        activation=None,
        #                        kernel_regularizer=regularizers.l2(kernel_l2_reg),
        #                        name='actions')(net)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=1e-5,
                                    # clipvalue=0.5,
                                    # clipnorm=1.0
                                    )

        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        kernel_l2_reg = 1e-5

        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # size_repeat = 20
        # state_size = size_repeat*self.state_size
        # action_size = size_repeat*self.action_size
        # block_size = size_repeat*self.state_size + size_repeat*self.action_size
        # print("Critic block size = {}".format(block_size))

        # net_states = layers.Dense(state_net_size,
        #                           kernel_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1),
        #                           bias_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
        #                           activation='tanh')(states)
        # net_states = layers.BatchNormalization()(net_states)
        #
        # net_actions = layers.Dense(action_net_size,
        #                            kernel_initializer=initializers.RandomNormal(mean=1.0, stddev=0.1),
        #                            bias_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
        #                            activation='tanh')(actions)
        # net_actions = layers.BatchNormalization()(net_actions)
        #
        # net = layers.concatenate([net_states, net_actions])

        # net = layers.concatenate([states]*size_repeat + [actions]*size_repeat)

        # net = layers.concatenate([states, actions])
        # net = layers.Dense(32, activation='relu')(net)

        # state_net = layers.concatenate([states] * size_repeat)
        # action_net = layers.concatenate([actions]*size_repeat)
        # for _ in range(2):
        #     state_net = res_block(state_net, state_size)
        #     action_net = res_block(action_net, action_size)
        #
        # net = layers.concatenate([state_net, action_net])
        #
        # for _ in range(5):
        #     # net = res_block(net, self.state_size + self.action_size)
        #     net = res_block(net, block_size)
        #     # net = layers.BatchNormalization()(net)
        #     # net = layers.Dropout(0.2)(net)


        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=300,
                                  # activation='relu',
                                  activation=None,
                                  # kernel_initializer='lecun_normal',
                                  # activity_regularizer=regularizers.l2(0.1),
                                  kernel_regularizer=regularizers.l2(kernel_l2_reg),
                                  # bias_initializer=initializers.Constant(1e-2),
                                  use_bias=True)(states)
        # net_states = layers.BatchNormalization()(net_states)
        net_states = layers.LeakyReLU(1e-2)(net_states)
        # net_states = layers.ELU()(net_states)

        net_states = layers.Dense(units=400,
                                  # activation='relu',
                                  activation=None,
                                  # kernel_initializer='lecun_normal',
                                  # activity_regularizer=regularizers.l2(0.01),
                                  kernel_regularizer=regularizers.l2(kernel_l2_reg),
                                  # bias_initializer=initializers.Constant(1e-2),
                                  use_bias=True)(net_states)
        # net_states = layers.BatchNormalization()(net_states)
        net_states = layers.LeakyReLU(1e-2)(net_states)
        # net_states = layers.ELU()(net_states)

        # net_states = layers.Dense(units=128,
        #                           activation='relu',
                                  # activation=None,
                                  # kernel_regularizer=regularizers.l2(kernel_l2_reg),
                                  # use_bias=True)(net_states)
        # net_states = layers.BatchNormalization()(net_states)
        # net_states = layers.LeakyReLU()(net_states)
        # net_states = layers.ELU()(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=400,
                                   # activation='relu',
                                   activation=None,
                                   # kernel_initializer='lecun_normal',
                                   # activity_regularizer=regularizers.l2(0.1),
                                   kernel_regularizer=regularizers.l2(kernel_l2_reg),
                                   # bias_initializer=initializers.Constant(1e-2),
                                   use_bias=False)(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.LeakyReLU(1e-2)(net_actions)
        # net_actions = layers.ELU()(net_actions)

        # net_actions = layers.Dense(units=64,
        #                            activation='relu',
                                   # activation=None,
                                   # activity_regularizer=regularizers.l2(0.01),
                                   # kernel_regularizer=regularizers.l2(kernel_l2_reg),
                                   # use_bias=True)(net_actions)
        # net_actions = layers.BatchNormalization()(net_actions)
        # net_actions = layers.LeakyReLU()(net_actions)
        # net_actions = layers.ELU()(net_actions)

        # net_actions = layers.Dense(units=128,
        #                            activation='relu',
                                   # activation=None,
                                   # kernel_regularizer=regularizers.l2(kernel_l2_reg),
                                   # use_bias=True)(net_actions)
        # net_actions = layers.BatchNormalization()(net_actions)
        # net_actions = layers.LeakyReLU()(net_actions)
        # net_actions = layers.ELU()(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        # net = layers.concatenate([net_states, net_actions])
        net = layers.add([net_states, net_actions])
        # net = layers.BatchNormalization()(net)
        # net = layers.ELU()(net)
        # net = layers.LeakyReLU(alpha=0.01)(net)
        # net = activations.selu(net)

        # net_actions = layers.ELU()(net_actions)

        # net = layers.Dense(units=300,
        #                    activation='relu',
                           # activation=None,
                           # activity_regularizer=regularizers.l2(0.1),
                           # kernel_regularizer=regularizers.l2(kernel_l2_reg),
                           # use_bias=True)(net)
        # net = layers.BatchNormalization()(net)
        # net = layers.LeakyReLU()(net)

        net = layers.Dense(units=32,
                           # activation='relu',
                           activation=None,
                           # kernel_initializer='lecun_normal',
                           # activity_regularizer=regularizers.l2(0.01),
                           kernel_regularizer=regularizers.l2(kernel_l2_reg),
                           # bias_initializer=initializers.Constant(1e-2),
                           use_bias=True)(net)
        # net = layers.BatchNormalization()(net)
        net = layers.LeakyReLU(1e-2)(net)
        # net = layers.ELU()(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1,
                                activation=None,
                                kernel_regularizer=regularizers.l2(kernel_l2_reg),
                                kernel_initializer=initializers.RandomUniform(minval=-5e-3, maxval=5e-3),
                                # bias_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=1e-5,
                                    # clipvalue=0.5,
                                    # clipnorm=1.0
                                    )#, beta_1=0.5)

        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)


def res_block(inputs, size):
    kernel_l2_reg = 1e-5

    net = layers.Dense(size,
                       activation=None,
                       # kernel_regularizer=regularizers.l2(kernel_l2_reg)
                       # kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                       # bias_initializer=initializers.RandomNormal(mean=0.0, stddev=0.001),
                       )(inputs)
    # net = layers.BatchNormalization()(net)
    # net = layers.Dropout(0.2)(net)
    net = layers.LeakyReLU(1e-2)(net)

    net = layers.Dense(size,
                       activation=None,
                       # kernel_regularizer=regularizers.l2(kernel_l2_reg)
                       # kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                       # bias_initializer=initializers.RandomNormal(mean=0.0, stddev=0.001),
                       )(net)
    # net = layers.BatchNormalization()(net)
    # net = layers.Dropout(0.2)(net)
    net = layers.add([inputs, net])
    # net = layers.BatchNormalization()(net)
    net = layers.LeakyReLU(1e-2)(net)
    return net


# class ReplayBuffer:
#     """Fixed-size buffer to store experience tuples."""
#
#     def __init__(self, buffer_size, batch_size):
#         """Initialize a ReplayBuffer object.
#         Params
#         ======
#             buffer_size: maximum size of buffer
#             batch_size: size of each training batch
#         """
#         self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#
#     def add(self, state, action, reward, next_state, done):
#         """Add a new experience to memory."""
#         if np.all([state, action, reward, next_state, done] is not None):
#             e = self.experience(state, action, reward, next_state, done)
#             self.memory.append(e)
#
#     def sample(self):
#         """Randomly sample a batch of experiences from memory."""
#         # return random.sample(self.memory, k=self.batch_size)
#         self.batch_index = np.random.choice(np.arange(len(self.memory)), self.batch_size, replace=False)
#
#         states = np.vstack([self.memory[i].state for i in self.batch_index])
#         actions = np.array([self.memory[i].action for i in self.batch_index]).astype(np.float32).reshape(self.batch_size, -1)
#         rewards = np.array([self.memory[i].reward for i in self.batch_index]).astype(np.float32).reshape(-1, 1)
#         next_states = np.vstack([self.memory[i].next_state for i in self.batch_index])
#         dones = np.array([self.memory[i].done for i in self.batch_index]).astype(np.uint8).reshape(-1, 1)
#
#         return states, actions, rewards, next_states, dones
#         # return self.memory[self.batch_index]
#
#     def __len__(self):
#         """Return the current size of internal memory."""
#         return len(self.memory)


# class ReplayBuffer:
#     """Fixed-size buffer to store experience tuples."""
#     def __init__(self, buffer_size, batch_size):
#         """Initialize a ReplayBuffer object.
#         Params
#         ======
#             buffer_size: maximum size of buffer
#             batch_size: size of each training batch
#         """
#         self.buffer_size = buffer_size
#         self.memory = np.empty(buffer_size, dtype=object)
#         self.p = np.zeros(buffer_size)
#         self.next_index = 0
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#
#         # Normalizer placeholders for state, action and reward.
#         self.state_norm = None
#         # self.action_norm = None
#         # self.reward_norm = None
#
#     def add(self, state, action, reward, next_state, done):
#         """Add a new experience to memory."""
#         if np.all([state, action, reward, next_state, done] is not None):
#
#             # Build normalizers on first data sample
#             # if self.state_norm is None:
#             #     self.state_norm = Normalizer(shape=state.shape, dtype=state.dtype)
#
#             # if self.action_norm is None:
#             #     self.action_norm = Normalizer(shape=action.shape, dtype=action.dtype)
#             #
#             # if self.reward_norm is None:
#             #     self.reward_norm = Normalizer(shape=reward.shape, dtype=reward.dtype)
#
#             # Update our normalizer
#             # self.state_norm.update(state)
#             # self.action_norm.update(action)
#             # self.reward_norm.update(reward)
#             # self.state_norm.update(next_state)
#
#             # Build experience and add to ring buffer
#             e = self.experience(state, action, reward, next_state, done)
#             self.memory[self.next_index] = e
#             self.p[self.next_index] = 1
#             self.next_index = (self.next_index + 1)%self.buffer_size
#
#     def sample(self):
#         """Randomly sample a batch of experiences from memory."""
#         # return random.sample(self.memory, k=self.batch_size)
#         batch = random.sample(list(self.memory[self.p == 1]), self.batch_size)
#
#         states = np.vstack([e.state for e in batch])
#         actions = np.array([e.action for e in batch]).astype(np.float32).reshape(self.batch_size, -1)
#         rewards = np.array([e.reward for e in batch]).astype(np.float32).reshape(-1, 1)
#         next_states = np.vstack([e.next_state for e in batch])
#         dones = np.array([e.done for e in batch]).astype(np.uint8).reshape(-1, 1)
#
#         # Normalize if possible
#         # if self.state_norm is not None:
#         #     states = self.state_norm.normalize(states)
#         #     next_states = self.state_norm.normalize(next_states)
#
#         # if self.action_norm is not None:
#         #     actions = self.action_norm.normalize(actions)
#         #
#         # if self.reward_norm is not None:
#         #     rewards = self.reward_norm.normalize(rewards)
#
#         return states, actions, rewards, next_states, dones
#
#     def __len__(self):
#         """Return the current size of internal memory."""
#         return int(sum(self.p))


# class Normalizer:
#     """
#     Online normalizer to normalize array to 0 mean and unit variance.
#
#     Based on the work here:
#     https://github.com/keras-rl/keras-rl/blob/master/rl/util.py
#     """
#     def __init__(self, shape, dtype):
#         self.shape = shape
#         self.dtype = dtype
#         self.epsilon = 1e-2 ** 2
#
#         self.mean = np.zeros(shape, dtype=dtype)
#         self.std = np.ones(shape, dtype=dtype)
#
#         self.sums = np.zeros(shape, dtype=dtype)
#         self.sqrs = np.zeros(shape, dtype=dtype)
#         self.n = 0
#
#     def update(self, x):
#         self.n += 1
#         self.sums += x
#         self.sqrs += x*x
#
#         self.mean = self.sums/float(self.n)
#         self.std = np.sqrt(np.maximum(self.epsilon, self.sqrs / float(self.n) - self.mean*self.mean))
#
#     def normalize(self, x):
#         return (x - self.mean)/self.std
#
#     def denormalize(self, x):
#         return x*self.std + self.mean


# class OUNoise:
#     """Ornstein-Uhlenbeck process."""
#
#     def __init__(self, size, mu, theta, sigma):
#         """Initialize parameters and noise process."""
#         self.mu = mu * np.ones(size)
#         self.theta = theta
#         self.sigma = sigma
#         self.reset()
#
#     def reset(self):
#         """Reset the internal state (= noise) to mean (mu)."""
#         self.state = self.mu
#
#     def sample(self):
#         """Update internal state and return it as a noise sample."""
#         x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
#         self.state = x + dx
#         return self.state
