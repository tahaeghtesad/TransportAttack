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
        self.truncateds = deque(maxlen=buffer_size)

    def add(self, state, allocation, budget, action, reward, next_state, done, truncated):
        self.states.append(state)
        self.allocations.append(allocation)
        self.budgets.append(budget)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.truncateds.append(truncated)

    def batch_add(self, experiences):
        for e in experiences:
            self.add(**e)

    def size(self):
        return len(self.states)

    def reset(self):
        pass

    def clear(self):
        self.states.clear()
        self.allocations.clear()
        self.budgets.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.truncateds.clear()

    def get_experiences(self):
            indices = random.choices(range(self.size()), k=self.batch_size)

            return (
                [self.states[i] for i in indices],
                [self.allocations[i] for i in indices],
                [self.budgets[i] for i in indices],
                [self.actions[i] for i in indices],
                [self.rewards[i] for i in indices],
                [self.next_states[i] for i in indices],
                [self.dones[i] for i in indices],
                [self.truncateds[i] for i in indices],
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


class BasicTrajectoryExperience:

    def __init__(self):
        super().__init__()
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.truncateds = []

    def add(self, state, action, reward, done, truncated):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.truncateds.append(truncated)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.truncateds = []

    def size(self):
        return len(self.states)

    def get_experiences(self):
        return (
            self.states,
            self.actions,
            self.rewards,
            self.dones,
            self.truncateds,
        )


class BasicExperienceReplay:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.next_states = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        self.truncateds = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done, truncated):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.truncateds.append(truncated)

    def batch_add(self, experiences):
        for e in experiences:
            self.add(**e)

    def size(self):
        return len(self.states)

    def reset(self):
        pass

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.truncateds.clear()

    def get_experiences(self) -> tuple[list, list, list, list, list, list]:
            indices = random.choices(range(self.size()), k=self.batch_size)

            return (
                [self.states[i] for i in indices],
                [self.actions[i] for i in indices],
                [self.rewards[i] for i in indices],
                [self.next_states[i] for i in indices],
                [self.dones[i] for i in indices],
                [self.truncateds[i] for i in indices],
            )
