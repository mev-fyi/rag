#!/bin/bash

# Image name
IMAGE_NAME="my-flask-app"

# Find container ID based on the image name
CONTAINER_ID=$(docker ps -qf "ancestor=$IMAGE_NAME")

if [ -n "$CONTAINER_ID" ]; then
    # Stop the Docker container
    echo "Stopping Flask app container..."
    docker stop "$CONTAINER_ID"

    # Remove the Docker container
    echo "Removing Flask app container..."
    docker rm "$CONTAINER_ID"
else
    echo "No running container found for image $IMAGE_NAME"
fi

# Attempt to remove the Docker image
echo "Removing Flask app image..."
docker rmi "$IMAGE_NAME" || echo "Image not found or unable to remove. It may be used by stopped containers."

# Build the new Docker image
echo "Building new Flask app image..."
docker build -t my-flask-app -f Dockerfile_app .

echo "Flask app image build complete!"

echo "Now launching Flask app on localhost:8080!"
docker run -it -p 8080:8080 --name my-flask-container my-flask-app /bin/bash
# docker run -it -p 8080:8080 --name my-flask-container my-flask-app
# Instead of running the container in detached mode (-d), run it interactively with -it and without the -d. This might allow you to see the output directly in your terminal.
# docker run -it -p 8080:8080 --name my-flask-container my-flask-app /bin/bash
