import logging
import math
import random


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
    def __init__(self, noise_start, noise_end, noise_decay):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.noise_start = noise_start
        self.noise_end = noise_end
        self.noise_decay = noise_decay
        self.step = 0

    def __call__(self):
        self.step += 1
        return self.get_current_noise()

    def get_current_noise(self):
        return self.noise_end + (self.noise_start - self.noise_end) * math.exp(
            -1. * self.step / self.noise_decay)
