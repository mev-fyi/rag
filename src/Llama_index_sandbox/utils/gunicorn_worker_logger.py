import logging

from flask import has_request_context, request

# Custom log format that includes contextual information when within a request
class RequestFormatter(logging.Formatter):
    def format(self, record):
        if has_request_context():
            record.url = request.url
            record.remote_addr = request.remote_addr
        else:
            record.url = None
            record.remote_addr = None
        return super().format(record)

# Use the custom formatter in the logging configuration
gunicorn_handler = logging.StreamHandler()
gunicorn_handler.setFormatter(RequestFormatter(
    '[%(asctime)s] %(remote_addr)s requested %(url)s\n'
    '%(levelname)s in %(module)s: %(message)s'
))

