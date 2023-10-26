import threading
import time
from collections import deque

class TokenCounter:
    def __init__(self, limit_per_minute):
        self.token_queue = deque()  # Queue to hold batches of token counts with their timestamp
        self.limit_per_minute = limit_per_minute
        self.lock = threading.Lock()

    def add(self, count):
        with self.lock:
            current_time = time.time()
            # Instead of adding a new timestamp for each token,
            # you add a single record for all tokens in this request.
            self.token_queue.append((current_time, count))

    def clear_old_tokens(self):
        current_time = time.time()

        # Remove the elements from the deque until we find one that is less than 60 seconds old.
        # This approach does not use slicing but removes elements from the start one by one, which is supported by deque.
        while self.token_queue and self.token_queue[0][0] < current_time - 60:
            self.token_queue.popleft()  # This removes the oldest elements until the condition is met

    def is_rate_limit_exceeded(self):
        with self.lock:
            self.clear_old_tokens()
            # Sum the tokens in the time window and check against the limit.
            total_tokens = sum(count for _, count in self.token_queue)
            return total_tokens >= self.limit_per_minute