# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Set the Python path to include the src directory
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Set environment variables
ENV PINECONE_API_KEY=sk-6a39da3a-8f05-4fc0-a9d0-06301c4aa4ef \
    PINECONE_API_ENVIRONMENT=gcp-starter \
    EMBEDDING_MODEL_NAME_OSS=BAAI/bge-large-en-v1.5 \
    LLM_MODEL_NAME_OPENAI=gpt-3.5-turbo \
    OPENAI_API_KEY=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 \
    SERVICE_ACCOUNT_FILE=your_service_account_file \
    YOUTUBE_API_KEY=your_youtube_api_key \
    YOUTUBE_CHANNELS=your_youtube_channels \
    YOUTUBE_PLAYLISTS=your_youtube_playlists \
    ASSEMBLY_AI_API_KEY=your_assembly_ai_api_key

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "src/Llama_index_sandbox/app.py"]
