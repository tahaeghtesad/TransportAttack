import time
import logging

timer_stats = dict()


class Timer(object):
    def __init__(self, name: 'str'):
        self.logger = logging.getLogger(f'Timer - {name}')
        self.name = name

        if name not in timer_stats:
            timer_stats[name] = [0, 0]

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        time_taken = time.time() - self.tstart
        timer_stats[self.name][0] += time_taken
        timer_stats[self.name][1] += 1
        if time_taken < 0.001:
            self.logger.debug(f'This iteration took {time_taken * 1000000:.0f}(\u03BCs).')
        elif time_taken < 1:
            self.logger.debug(f'This iteration took {time_taken * 1000:.3f}(ms).')
        elif 1 <= time_taken <= 60:
            self.logger.debug(f'This iteration took {time_taken:.3f}(s).')
        elif time_taken > 3600:
            self.logger.debug(f'This iteration took {time_taken / 3600:.0f}(h).')
        elif time_taken > 60:
            self.logger.debug(f'This iteration took {time_taken / 60:.0f}(m).')

    @staticmethod
    def report(name):
        logging.getLogger(f'Timer Report - {name}').info(f'Total time: {timer_stats[name][0]:.3f}(s) - Average time: {timer_stats[name][0] / timer_stats[name][1]:.3f}(s) - Iterations: {timer_stats[name][1]}')

    @staticmethod
    def format(name):
        return f'Total time: {timer_stats[name][0]:.3f}(s) - Average time: {timer_stats[name][0] / timer_stats[name][1]:.3f}(s) - Iterations: {timer_stats[name][1]}'
