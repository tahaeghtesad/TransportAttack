import logging
import math
import random

import numpy as np

class DecayEpsilon:
    def __init__(self, epsilon_start, epsilon_end, epsilon_decay):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step = 0

    def reset(self):
        self.step = 0

    def __call__(self):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * self.step / self.epsilon_decay)
        self.step += 1
        return random.random() < epsilon

    def get_current_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * self.step / self.epsilon_decay)


class ConstantEpsilon:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self):
        return random.random() < self.epsilon

    def get_current_epsilon(self):
        return self.epsilon


class NoiseDecay:
    def __init__(self, noise_start, noise_end, noise_decay, shape):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.noise_start = noise_start
        self.noise_end = noise_end
        self.noise_decay = noise_decay
        self.shape = shape
        self.step = 0

    def __call__(self):
        self.step += 1
        return np.random.normal(0, self.get_current_noise(), self.shape)

    def get_current_noise(self):
        return self.noise_end + (self.noise_start - self.noise_end) * math.exp(
            -1. * self.step / self.noise_decay)


class ZeroNoise:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self):
        return np.zeros(self.shape)

    def get_current_noise(self):
        return 0.0


class OUActionNoise:
    def __init__(self, mean, std_deviation, shape, target_scale, anneal, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.shape = shape
        self.x_prev = None
        self.target_scale = target_scale
        self.scale = 0
        self.anneal = anneal
        self.step_count = 0
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x

        self.step_count += 1
        self.scale = self.target_scale + (1.0 - self.target_scale) * math.exp(
            -1. * self.step_count / self.anneal)

        return x * self.scale

    def reset(self):
        self.step_count = 0
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros(self.shape)

    def get_current_noise(self):
        return self.scale * self.std_dev
