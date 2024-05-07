import numpy as np
from torch.utils import tensorboard as tb


class TBStatWriter:
    def __init__(self, path):
        self.writer = tb.SummaryWriter(path)

    def add_stats(self, stats, step):
        for name, value in stats.items():
            if type(value) is list:
                self.writer.add_histogram(name, np.array(value), step)
            elif type(value) is np.ndarray:
                self.writer.add_histogram(name, value, step)
            else:
                self.writer.add_scalar(name, value, step)

    def add_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)

    def add_histogram(self, name, value, step):
        self.writer.add_histogram(name, value, step)
