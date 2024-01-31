#!/bin/bash

# Install Docker (if not already installed)
sudo apt-get update
sudo apt-get install -y docker.io

# Pull the Docker image
docker pull gcr.io/mev-fyi/twitter-bot-app:$SHORT_SHA

# Run the Docker container
docker run -d --name my_twitter_bot_container gcr.io/mev-fyi/twitter-bot-app:$SHORT_SHA
