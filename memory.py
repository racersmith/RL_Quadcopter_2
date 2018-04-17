import numpy as np
from collections import deque, namedtuple
import random


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if np.all([state, action, reward, next_state, done] is not None):
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # return random.sample(self.memory, k=self.batch_size)
        batch = random.sample(list(self.memory), self.batch_size)

        states = np.vstack([e.state for e in batch])
        actions = np.array([e.action for e in batch]).astype(np.float32).reshape(self.batch_size, -1)
        rewards = np.array([e.reward for e in batch]).astype(np.float32).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in batch])
        dones = np.array([e.done for e in batch]).astype(np.uint8).reshape(-1, 1)

        return states, actions, rewards, next_states, dones
        # return self.memory[self.batch_index]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class RingBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.buffer_size = buffer_size
        self.memory = [None]*buffer_size
        # self.p = np.zeros(buffer_size)
        self.next_index = 0
        self.size = 0
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        # Normalizer placeholders for state, action and reward.
        self.state_norm = None
        # self.action_norm = None
        # self.reward_norm = None

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if np.all([state, action, reward, next_state, done] is not None):

            # Build normalizers on first data sample
            # if self.state_norm is None:
            #     self.state_norm = Normalizer(shape=state.shape, dtype=state.dtype)

            # if self.action_norm is None:
            #     self.action_norm = Normalizer(shape=action.shape, dtype=action.dtype)
            #
            # if self.reward_norm is None:
            #     self.reward_norm = Normalizer(shape=reward.shape, dtype=reward.dtype)

            # Update our normalizer
            # self.state_norm.update(state)
            # self.action_norm.update(action)
            # self.reward_norm.update(reward)
            # self.state_norm.update(next_state)

            # Build experience and add to ring buffer
            e = self.experience(state, action, reward, next_state, done)
            self.memory[self.next_index] = e

            # Increment counts
            self.next_index = self.next_index + 1
            self.size = max(self.size, self.next_index)
            if self.next_index >= self.buffer_size:
                self.next_index = 0

    def sample(self, normalize=True):
        # Randomly sample a batch from memory
        batch = random.sample(self.memory[:self.size], self.batch_size)

        # Structure data for learning
        states = np.vstack([e.state for e in batch])
        actions = np.array([e.action for e in batch]).astype(np.float32).reshape(self.batch_size, -1)
        rewards = np.array([e.reward for e in batch]).astype(np.float32).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in batch])
        dones = np.array([e.done for e in batch]).astype(np.uint8).reshape(-1, 1)

        # Normalize if possible
        # if normalize and self.state_norm is not None:
        #     states = self.state_norm.normalize(states)
        #     next_states = self.state_norm.normalize(next_states)

        # if self.action_norm is not None:
        #     actions = self.action_norm.normalize(actions)
        #
        # if self.reward_norm is not None:
        #     rewards = self.reward_norm.normalize(rewards)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size


class Normalizer:
    """
    Online normalizer to normalize array to 0 mean and unit variance.

    Based on the work here:
    https://github.com/keras-rl/keras-rl/blob/master/rl/util.py
    """
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.epsilon = 1e-2 ** 2

        self.mean = np.zeros(shape, dtype=dtype)
        self.std = np.ones(shape, dtype=dtype)

        self.sums = np.zeros(shape, dtype=dtype)
        self.sqrs = np.zeros(shape, dtype=dtype)
        self.n = 0

    def update(self, x):
        self.n += 1
        self.sums += x
        self.sqrs += x*x

        self.mean = self.sums/float(self.n)
        self.std = np.sqrt(np.maximum(self.epsilon, self.sqrs / float(self.n) - self.mean*self.mean))

    def normalize(self, x):
        return (x - self.mean)/self.std

    def denormalize(self, x):
        return x*self.std + self.mean
