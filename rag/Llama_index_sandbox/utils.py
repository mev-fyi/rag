import multiprocessing
import time
import random
import logging
import os


def root_directory() -> str:
    """
    Determine the root directory of the project based on the presence of '.git' directory.

    Returns:
    - str: The path to the root directory of the project.
    """
    current_dir = os.getcwd()

    while True:
        if '.git' in os.listdir(current_dir):
            return current_dir
        else:
            # Go up one level
            current_dir = os.path.dirname(current_dir)


class RateLimitController:
    def __init__(self):
        self.backoff_time = multiprocessing.Value('d', 20.0)
        self.lock = multiprocessing.Lock()

    def register_rate_limit_exceeded(self):
        with self.lock:
            jitter = self.backoff_time.value * 0.5 * random.uniform(0, 1)
            sleep_time = min(self.backoff_time.value + jitter, 300)
            logging.warning(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
            self.backoff_time.value = min(self.backoff_time.value * 2, 300)

    def reset_backoff_time(self):
        with self.lock:
            self.backoff_time.value = 5.0