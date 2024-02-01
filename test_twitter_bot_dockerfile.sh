#!/bin/bash

# Image name
IMAGE_NAME="twitter-bot-app"

# Assuming the script is in the same directory as the Dockerfile
DOCKERFILE_PATH="$(dirname "$0")/Dockerfile_twitter_bot_app"
echo "Dockerfile path: $DOCKERFILE_PATH"

# Check if the Dockerfile exists
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "Dockerfile not found at $DOCKERFILE_PATH"
    exit 1
fi

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
DANGLING_IMAGES=$(docker images -f "dangling=true" -q)
if [ -n "$DANGLING_IMAGES" ]; then
    echo "Removing dangling images..."
    docker rmi $DANGLING_IMAGES
else
    echo "No dangling images to remove."
fi

# Pruning all stopped containers, unused networks, and build cache
echo "Pruning containers, networks, and build cache..."
docker system prune -af --volumes

# Build the new Docker image
echo "Building new Twitter Bot app image..."
docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH .

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "Failed to build the image $IMAGE_NAME."
    exit 1
else
    echo "Twitter Bot app image build complete!"
fi

# Run the new Docker container and log its output
echo "Starting Twitter Bot app container..."
docker run -d --name "${IMAGE_NAME}-container" $IMAGE_NAME > container_output.log 2>&1

if [ $? -eq 0 ]; then
    echo "Twitter Bot app container started. Container ID: ${IMAGE_NAME}-container"
else
    echo "Failed to start the Twitter Bot app container."
fi

# Display container logs
echo "Container logs:"
docker logs "${IMAGE_NAME}-container"