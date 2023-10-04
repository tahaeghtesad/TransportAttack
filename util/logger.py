import logging
import os
from datetime import datetime


class CustomFormatter(logging.Formatter):
    red = "\x1b[31;21m"
    green = "\x1b[32;21m"
    yellow = "\x1b[33;21m"
    blue = "\x1b[34;21m"
    magenta = "\x1b[35;21m"
    cyan = "\x1b[36;21m"
    bright_red = "\x1b[91;21m"
    orange = "\x1b[93;21m"
    reset = "\x1b[0m"
    format = "%(message)s"
    date_fmt = '%Y-%m-%d %H:%M:%S'
    time_fmt = cyan + "[%(asctime)s] "
    level_fmt = "[%(levelname)s] "

    FORMATS = {
        logging.INFO: time_fmt + blue + level_fmt + reset + format,
        logging.DEBUG: time_fmt + yellow + level_fmt + reset + format,
        logging.WARNING: time_fmt + orange + level_fmt + reset + format,
        logging.ERROR: time_fmt + red + level_fmt + reset + format,
        logging.CRITICAL: time_fmt + bright_red + level_fmt + reset + format,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.date_fmt)
        return formatter.format(record)


class LogFormatter(logging.Formatter):
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    date_fmt = '%Y-%m-%d %H:%M:%S'

    def format(self, record):
        formatter = logging.Formatter(self.fmt, datefmt=self.date_fmt)
        return formatter.format(record)


logger = logging.getLogger("EMS")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = CustomFormatter()
ch.setFormatter(ch_formatter)
logger.addHandler(ch)
try:
    os.mkdir("logs")
except FileExistsError:
    pass
file_handler = logging.FileHandler(f'logs/ems-{int(datetime.now().timestamp())}.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(LogFormatter())
logger.addHandler(file_handler)
