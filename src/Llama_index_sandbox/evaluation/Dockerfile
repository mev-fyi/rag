FROM python:3.9-slim

WORKDIR /app

# You might have different dependencies for the evaluation script
COPY src/Llama_index_sandbox/evaluation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the evaluation directory
COPY src/Llama_index_sandbox/evaluation .

# Command to run the evaluation script
CMD ["python", "eval.py"]