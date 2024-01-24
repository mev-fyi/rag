#!/bin/bash

# Define the Docker image name
IMAGE_NAME="my-twitter-bot"

# Path to the Dockerfile
DOCKERFILE_PATH="Dockerfile_twitter_bot_app"

# Cleanup any previous failed script
echo "Cleanup container from previous script..."
docker stop twitter_bot_container
docker rm twitter_bot_container

# Build the Docker image
echo "Building Twitter Bot Docker image..."
docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH .

# Run the Docker container
echo "Running Twitter Bot Docker container..."
docker run -d --name twitter_bot_container $IMAGE_NAME

# Wait for the container to start
echo "Waiting for the container to start..."
sleep 10

# Run the simulation script inside the container
echo "Simulating a webhook event..."
docker exec twitter_bot_container python -c "
from src.Llama_index_sandbox.twitter_bot import TwitterBot
bot = TwitterBot()
bot.simulate_webhook_event(user_id='123456', tweet_id='654321', tweet_text='Hello, Twitter Bot!', command_type='tweet')
"

# Show logs for debugging
echo "Docker logs for Twitter Bot:"
docker logs twitter_bot_container

# Cleanup
echo "Stopping and removing the container..."
docker stop twitter_bot_container
docker rm twitter_bot_container

echo "Test script completed."
