FROM python:3.9-slim

WORKDIR /app

# Copy the requirements file to the working directory in the container
COPY src/Llama_index_sandbox/evaluation/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the evaluation script directory
COPY src/Llama_index_sandbox/evaluation .

# Command to run the evaluation script
CMD ["python", "eval.py"]