import logging
import multiprocessing
import random
import time
from functools import wraps


class RateLimitController:
    """
    A controller to handle rate limiting by implementing a backoff mechanism.

    Attributes:
    - backoff_time (multiprocessing.Value): Shared value to store the current backoff time.
    - lock (multiprocessing.Lock): Lock to ensure thread-safety when updating the backoff time.

    Methods:
    - register_rate_limit_exceeded: Increases the backoff time and sleeps the process for that duration.
    - reset_backoff_time: Resets the backoff time to the default value.
    """

    def __init__(self):
        self.backoff_time = multiprocessing.Value('d', 20.0)
        self.lock = multiprocessing.Lock()
        self.pause_event = multiprocessing.Event()  # Event flag to control pausing across threads

    def register_rate_limit_exceeded(self):
        with self.lock:
            try:
                self.pause_event.set()
                jitter = self.backoff_time.value * 0.5 * random.uniform(0, 1)
                sleep_time = min(self.backoff_time.value + jitter, 300)
                logging.warning(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                self.backoff_time.value = min(self.backoff_time.value * 2, 300)
                logging.info("Clearing pause event...")  # New log statement
            except Exception as e:
                logging.exception("An error occurred during backoff: %s", e)
            finally:
                self.pause_event.clear()  # Moved to 'finally' to ensure it executes
                logging.info("Pause event cleared.")

    def wait_if_paused(self):
        while self.pause_event.is_set():
            logging.info("Rate limit controller paused. Waiting for the pause to be lifted...")
            paused = self.pause_event.wait(timeout=30)  # Wait for a maximum of 60 seconds
            if not paused:
                logging.warning("Waited for the pause to be lifted for 30 seconds, checking again...")

    def reset_backoff_time(self):
        """
        Resets the backoff time to the default value of 40.0 seconds.
        """
        with self.lock:
            self.backoff_time.value = 20.0


rate_limit_controller = RateLimitController()


def handle_rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            rate_limit_controller.wait_if_paused()  # Uses the global instance
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if 'Rate limit reached' in str(e) or 'rate_limit_exceeded' in str(e):
                    logging.warning("Rate limit error detected.")
                    rate_limit_controller.register_rate_limit_exceeded()
                else:
                    logging.error(f"Failed due to: {e}")
                    raise  # Important to re-raise the exception
    return wrapper
