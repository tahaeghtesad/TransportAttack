import random
from collections import deque

import numpy as np


class ExperienceReplay:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.next_states = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        self.next_actions = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done, next_action):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.next_actions.append(next_action)

    def batch_add(self, experiences):
        for e in experiences:
            self.add(**e)

    def size(self):
        return len(self.states)

    def sample(self):
        indices = random.choices(range(self.size()), k=self.batch_size)
        return (
            np.array([self.states[i] for i in indices], dtype=np.float32),
            np.array([self.actions[i] for i in indices], dtype=np.float32),
            np.array([self.next_states[i] for i in indices], dtype=np.float32),
            np.array([self.rewards[i] for i in indices], dtype=np.float32),
            np.array([self.dones[i] for i in indices], dtype=np.float32),
            np.array([self.next_actions[i] for i in indices], dtype=np.float32),
        )
