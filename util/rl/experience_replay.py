import random
from collections import deque

import tensorflow as tf


class ExperienceReplay:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, obs, action, reward, next_obs, next_action, done):
        self.buffer.append(
            dict(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                next_action=next_action,
                done=done
            )
        )

    def batch_add(self, experiences):
        self.buffer.extend(experiences)

    def size(self):
        return len(self.buffer)

    def sample(self):
        ret = dict(
            states=[],
            actions=[],
            rewards=[],
            next_states=[],
            dones=[],
            next_actions=[]
        )

        experiences = random.choices(self.buffer, k=self.batch_size)

        for e in experiences:
            ret['states'].append(e['state'])
            ret['actions'].append(e['action'])
            ret['rewards'].append([e['reward']])
            ret['next_states'].append(e['next_state'])
            ret['dones'].append([e['done']])
            if 'next_action' in e:
                ret['next_actions'].append(e['next_action'])

        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in ret.items()}
