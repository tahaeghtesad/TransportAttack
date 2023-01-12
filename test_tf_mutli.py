from multiprocessing import Process, Pipe
from tqdm import tqdm

import tensorflow as tf
import numpy as np


class Agent(Process):
    def __init__(self, index, pipe) -> None:
        super().__init__(name=f'Agent-{index}')
        self.pipe = pipe
        self.index = index

    def run(self) -> None:
        gpus = tf.config.list_physical_devices('GPU')
        assigned_gpu = self.index % len(gpus)
        print(tf.config.get_visible_devices())
        tf.config.set_visible_devices(gpus[assigned_gpu], 'GPU')
        print(tf.config.get_visible_devices())
        tf.config.set_logical_device_configuration(
            gpus[assigned_gpu],
            [tf.config.LogicalDeviceConfiguration(memory_limit=512)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(5, 5)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        while True:
            data = self.pipe.recv()
            if data is None:
                break
            # self.pipe.send(model(data).numpy())
            self.pipe.send(np.zeros((1, 1)))


if __name__ == '__main__':

    # gpus = tf.config.list_physical_devices('GPU')
    # assigned_gpu = 0
    # tf.config.set_visible_devices(gpus[assigned_gpu], 'GPU')
    # tf.config.set_logical_device_configuration(
    #     gpus[assigned_gpu],
    #     [tf.config.LogicalDeviceConfiguration(memory_limit=512)])
    # logical_gpus = tf.config.list_logical_devices('GPU')
    # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    #
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(5, 5)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    pipes = []
    for i in range(4):
        parent, child = Pipe()
        agent = Agent(i, child)
        pipes.append(parent)
        agent.start()

    for i in tqdm(range(10)):
        for j, pipe in enumerate(pipes):
            print(f'Sent sample {i} to agent {j}')
            pipe.send(np.random.uniform(size=(128, 5, 5)))

    for pipe in pipes:
        for _ in tqdm(range(10)):
            pipe.recv()
        pipe.close()