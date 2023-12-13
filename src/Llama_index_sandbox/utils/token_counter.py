import threading
import time
from collections import deque

class TokenCounter:
    def __init__(self, limit_per_minute):
        self.token_queue = deque()  # Queue to hold token timestamps
        self.limit_per_minute = limit_per_minute
        self.lock = threading.Lock()

    def add(self, count):
        with self.lock:
            current_time = time.time()
            self.token_queue.append((current_time, count))
            self.clear_old_tokens()

    def clear_old_tokens(self):
        current_time = time.time()
        # Remove tokens older than 60 seconds
        while self.token_queue and self.token_queue[0][0] < current_time - 60:
            self.token_queue.popleft()

    def is_rate_limit_exceeded(self):
        with self.lock:
            self.clear_old_tokens()
            total_tokens = sum(count for _, count in self.token_queue)
            return total_tokens >= self.limit_per_minute