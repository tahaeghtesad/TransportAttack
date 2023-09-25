import random
from collections import deque

import numpy as np


class ExperienceReplay:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.states = deque(maxlen=buffer_size)
        self.allocations = deque(maxlen=buffer_size)
        self.budgets = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.next_states = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)

    def add(self, state, allocation, budget, action, reward, next_state, done):
        self.states.append(state)
        self.allocations.append(allocation)
        self.budgets.append(budget)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def batch_add(self, experiences):
        for e in experiences:
            self.add(**e)

    def size(self):
        return len(self.states)

    def reset(self):
        pass

    def get_experiences(self, n_batches):
        for _ in range(n_batches):
            indices = random.choices(range(self.size()), k=self.batch_size)

            yield (
                [self.states[i] for i in indices],
                [self.allocations[i] for i in indices],
                [self.budgets[i] for i in indices],
                [self.actions[i] for i in indices],
                [self.rewards[i] for i in indices],
                [self.next_states[i] for i in indices],
                [self.dones[i] for i in indices],
            )


class TrajectoryExperience:
    def __init__(self):
        super().__init__()
        self.batch_size = 1

        self.states = []
        self.allocations = []
        self.budgets = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.truncateds = []

    def add(self, state, allocation, budget, action, reward, next_state, done, truncated):
        self.states.append(state)
        self.allocations.append(allocation)
        self.budgets.append(budget)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.truncateds.append(truncated)

    def reset(self):
        self.states = []
        self.allocations = []
        self.budgets = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.truncateds = []

    def size(self):
        return len(self.states)

    def get_experiences(self, n_batches=None):

        return (
            self.states,
            self.allocations,
            self.budgets,
            self.actions,
            self.rewards,
            self.next_states,
            self.dones,
            self.truncateds,
        )
