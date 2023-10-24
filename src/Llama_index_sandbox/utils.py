import asyncio
import multiprocessing
import time
import random
import logging
import os
from datetime import datetime
from functools import wraps

from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials


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
        # Doubling the default backoff time to 40.0 seconds
        self.backoff_time = multiprocessing.Value('d', 40.0)
        self.lock = multiprocessing.Lock()

    def register_rate_limit_exceeded(self):
        """
        Registers that a rate limit has been exceeded. This method will add a jitter (random fraction)
        to the current backoff time and sleep for that duration. Subsequently, the backoff time will be doubled,
        but capped at 300 seconds (5 minutes).
        """
        with self.lock:
            jitter = self.backoff_time.value * 0.5 * random.uniform(0, 1)
            sleep_time = min(self.backoff_time.value + jitter, 300)
            logging.warning(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
            self.backoff_time.value = min(self.backoff_time.value * 2, 300)

    def reset_backoff_time(self):
        """
        Resets the backoff time to the default value of 40.0 seconds.
        """
        with self.lock:
            self.backoff_time.value = 40.0


def start_logging():
    # Ensure that root_directory() is defined and returns the path to the root directory

    # Create a 'logs' directory if it does not exist
    if not os.path.exists(f'{root_directory()}/logs/txt'):
        os.makedirs(f'{root_directory()}/logs/txt')

    # Get the current date and time
    now = datetime.now()
    timestamp_str = now.strftime('%Y-%m-%d_%H-%M')

    # Set up the logging level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Add handler to log messages to a file
    log_filename = f'{root_directory()}/logs/txt/log_{timestamp_str}.txt'
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

    # Add handler to log messages to the standard output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)

    # Now, any logging.info() call will append the log message to the specified file and the standard output.
    logging.info('********* LOGGING STARTED *********')


def timeit(func):
    """
    A decorator that logs the time a function takes to execute.

    Args:
        func (callable): The function being decorated.

    Returns:
        callable: The wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        The wrapper function to execute the decorated function and log its execution time.

        Args:
            *args: Variable length argument list to pass to the decorated function.
            **kwargs: Arbitrary keyword arguments to pass to the decorated function.

        Returns:
            The value returned by the decorated function.
        """
        logging.info(f"\n{func.__name__} STARTED.")
        start_time = time.time()

        # Call the decorated function and store its result.
        # *args and **kwargs are used to pass the arguments received by the wrapper
        # to the decorated function.
        result = func(*args, **kwargs)

        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        logging.info(f"{func.__name__} COMPLETED, took {int(minutes)} minutes and {seconds:.2f} seconds to run.\n")

        return result  # Return the result of the decorated function

    return wrapper


def background(f):
    """
    Decorator that turns a synchronous function into an asynchronous function by running it in an
    executor using the default event loop.

    Args:
        f (Callable): The function to be turned into an asynchronous function.

    Returns:
        Callable: The wrapped function that can be called asynchronously.
    """
    def wrapped(*args, **kwargs):
        """
        Wrapper function that calls the original function 'f' in an executor using the default event loop.

        Args:
            *args: Positional arguments to pass to the original function 'f'.
            **kwargs: Keyword arguments to pass to the original function 'f'.

        Returns:
            Any: The result of the original function 'f'.
        """
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


def authenticate_service_account(service_account_file: str) -> Credentials:
    """Authenticates using service account and returns the session."""

    credentials = ServiceAccountCredentials.from_service_account_file(
        service_account_file,
        scopes=["https://www.googleapis.com/auth/youtube.readonly"]
    )
    return credentials
