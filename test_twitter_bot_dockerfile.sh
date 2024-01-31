#!/bin/bash

# Image name
IMAGE_NAME="twitter-bot-app"

# Assuming the script is in the same directory as the Dockerfile
DOCKERFILE_PATH="$(dirname "$0")/Dockerfile"

# Find container ID based on the image name
CONTAINER_ID=$(docker ps -qf "ancestor=$IMAGE_NAME")

if [ -n "$CONTAINER_ID" ]; then
    # Stop the Docker container
    echo "Stopping Twitter Bot app container..."
    docker stop "$CONTAINER_ID"

    # Remove the Docker container
    echo "Removing Twitter Bot app container..."
    docker rm "$CONTAINER_ID"
else
    echo "No running container found for image $IMAGE_NAME"
fi

# Attempt to remove the Docker image forcefully
echo "Forcibly removing the $IMAGE_NAME image..."
docker rmi -f $IMAGE_NAME || echo "Failed to remove the image $IMAGE_NAME."

# Cleanup dangling images (those tagged as none)
echo "Removing dangling images..."
docker rmi $(docker images -f "dangling=true" -q) || echo "No dangling images to remove."

# Pruning all stopped containers, unused networks, and build cache
echo "Pruning containers, networks, and build cache..."
docker system prune -af --volumes

# Build the new Docker image
echo "Building new Twitter Bot app image..."
docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH .

echo "Twitter Bot app image build complete!"

# Run the new Docker container
echo "Starting Twitter Bot app container..."
docker run -d --name "${IMAGE_NAME}-container" $IMAGE_NAME

echo "Twitter Bot app container started. Container ID: ${IMAGE_NAME}-container"
