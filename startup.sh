#!/bin/bash

# Install Docker if not already installed
if ! command -v docker &> /dev/null
then
    sudo apt-get update
    sudo apt-get install -y docker.io
fi

# Authenticate Docker with GCR (Assumes gcloud is installed and configured)
gcloud auth configure-docker

# Pull the latest Docker image
docker pull gcr.io/mev-fyi/twitter-bot-app:latest

# Run the Docker container
docker run -d --name my_twitter_bot_container gcr.io/mev-fyi/twitter-bot-app:latest